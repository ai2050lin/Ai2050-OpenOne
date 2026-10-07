# -*- coding: utf-8 -*-
"""
Phase 14 / N2h1-alpha-7 : 逐层累积代换（prefix swap）—— 可行性探针（观测前）
================================================================================
目的（按 Phase 12 教训 #14「可行性探针前置到 seal 之前」）：
  在冻结任何设计之前，先验证三件事，全部只测【已存在的装置事实】，不产生新统计量：
    (P-A) token 布局：TMPL % w 的 token 数 T 是多少？末位是哪个 token？
          （prefix 替换的「positions[0..t]」必须基于真实 token 数，不能猜。）
    (P-B) 单次前向成本：估 prefix 版总开销，决定 arm 规模是否落在预算内。
    (P-C) 装置锚可达性：
          - FULL_SWAP 能否从 Phase 12 已落盘的 FULL_SWAP_pairs 重算并与 Phase 12 落盘值逐位相等？
          - mean||P_U6(diff6)|| 能否重算并与 Phase 9 参照 17.0613 在 tol 内？
          - α=0 的「全位点替换」是否与未干预前向逐位同分（F3 的 prefix 版本）？
          - α=1 的全位点替换是否精确等于供体自身前向（prefix 版的 F12）？
  本脚本【不写入任何 Phase 14 判决量】，只产出 _feas_probe.txt。
"""
import os, sys, io, json, time, hashlib
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P12T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase12')
P14T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase14')
os.makedirs(P14T, exist_ok=True)
OUT = os.path.join(P12T, '_feas_probe_phase14.txt')
if os.environ.get('P14_OUT_IN_TEMP', '1') == '1':
    OUT = os.path.join(P14T, '_feas_probe.txt')

lines = []
def w(s=''):
    lines.append(str(s)); print(s); sys.stdout.flush()

E12 = json.load(io.open(os.path.join(P12T, 'execution_phase12.json'), encoding='utf-8'))
R12 = json.load(io.open(os.path.join(P12T, 'result_phase12.json'), encoding='utf-8'))

w('=== Phase 14 feasibility probe (pre-seal, observation-free) ===')
w('time %s' % time.strftime('%Y-%m-%d %H:%M:%S'))

MODEL = E12['model']
MDIR = os.path.join(ROOT, 'models', 'hf', MODEL)
TMPL = E12['template']
SUP_ID = {k: int(v) for k, v in E12['sup_id'].items()}
SUPS = list(E12['classes'])
PRIMARY = int(E12['primary_layer'])
DISC = [tuple(x) for x in E12['discovery']]
CONF = [tuple(x) for x in E12['confirmation']]
PAIRS_ALL = [tuple(p) for p in E12['pairs_all']]
INST_ALL = [tuple(x) for x in E12['instances_all']]
PROFILE = list(E12['profile_sites'])

# ---------------------------------------------------------------- P-C(1)
w('')
w('--- (P-C1) FULL_SWAP 从 Phase 12 落盘 per-pair 重算 ---')
FSP = R12['FULL_SWAP_pairs']
FS_VEC = np.array([FSP[x] for x in R12['E2']['6'][0]['order']], float)
FS_REC = float(np.mean(FS_VEC))
w('recomputed FULL_SWAP = %.15f' % FS_REC)
w('Phase 12 landed FULL_SWAP = %.15f' % float(R12['FULL_SWAP']))
w('BIT-EQUAL = %s  | max|d| = %.3e' % (FS_REC == float(R12['FULL_SWAP']),
                                        abs(FS_REC - float(R12['FULL_SWAP']))))
w('n FULL_SWAP_pairs keys = %d ; n order = %d' % (len(FSP), len(R12['E2']['6'][0]['order'])))

# ---------------------------------------------------------------- torch 段
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MDIR, dtype=torch.bfloat16, trust_remote_code=True,
                                             attn_implementation='eager').to('cuda').eval()
_core = getattr(model.model, 'language_model', model.model)
layers = _core.layers
L = len(layers)
HID = model.config.hidden_size
CFG = model.config
NORM = None
for nm in ['norm', 'final_layernorm', 'ln_f']:
    if hasattr(_core, nm):
        NORM = getattr(_core, nm); break
if NORM is None:
    for nm in ['norm', 'final_layernorm', 'ln_f']:
        if hasattr(model.model, nm):
            NORM = getattr(model.model, nm); break
w('')
w('--- 装置事实 ---')
# 注：head_dim 必须【直读 config.head_dim】。qwen3-4b 是 GQA，hidden/n_heads 反推会得到错的 80。
_HDIR = int(getattr(CFG, 'head_dim', -1))
_NKV = int(getattr(CFG, 'num_key_value_heads', CFG.num_attention_heads))
_OPR = None
for _nm in ['o_proj', 'dense', 'out_proj']:
    if hasattr(layers[PRIMARY if hasattr(CFG, 'num_hidden_layers') else 0].self_attn, _nm):
        _OPR = getattr(layers[PRIMARY].self_attn, _nm); break
_OPI = int(_OPR.in_features) if _OPR is not None else -1
w('model=%s L=%d hid=%d heads=%d kv_heads=%d head_dim(config)=%d o_proj_in=%d tie=%s' %
  (MODEL, L, HID, CFG.num_attention_heads, _NKV, _HDIR, _OPI, CFG.tie_word_embeddings))
w('  [勘误] 用 hidden/n_heads 反推的 head_dim = %d —— 对 GQA 模型是错的，勿再使用' %
  (HID // CFG.num_attention_heads))
w('  is_gqa = %s (kv_heads %d != heads %d)' % (_NKV != CFG.num_attention_heads, _NKV, CFG.num_attention_heads))
w('load time %.1fs ; norm module = %s' % (time.time() - t0, type(NORM).__name__))

def ids_of(s):
    return tok.encode(s, add_special_tokens=False)

w('')
w('--- (P-A) token 布局（TMPL = %r）---' % TMPL)
TOKS, TIDS = {}, {}
for wd, sup in INST_ALL:
    t = tok.encode(TMPL % wd, add_special_tokens=False)
    TOKS[wd] = [tok.decode([i]) for i in t]
    TIDS[wd] = t
uniq_T = sorted(set(len(v) for v in TOKS.values()))
w('distinct token-counts across 41 instances = %s' % uniq_T)
show = [INST_ALL[0][0], '葡萄', '香蕉', '红', '白', '银']
for wd in show:
    if wd in TOKS:
        w('  %-4s -> T=%d  %s' % (wd, len(TIDS[wd]), TOKS[wd]))
maxT = max(len(v) for v in TIDS.values())
w('T_MAX = %d ; 前缀替换的最大位置数 = %d' % (maxT, maxT))
w('instance self token id (first token) 与 class sup_id 是否重叠: %s' %
  {wd: (TIDS[wd][0], SUP_ID[TOKS[wd][0]] if TOKS[wd][0] in SUP_ID else None)
   for wd, _ in INST_ALL[:6]})

# ---------------------------------------------------------------- 捕获
@torch.no_grad()
def capture(text):
    ii = torch.tensor([ids_of(text)], device='cuda')
    rec = {}
    def hk(mod, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        rec['hR'] = t[0, -1].float().detach().cpu().numpy()
        return out
    h = NORM.register_forward_hook(hk)
    try:
        o = model(input_ids=ii, output_hidden_states=True)
    finally:
        h.remove()
    HH = np.stack([x[0].float().detach().cpu().numpy() for x in o.hidden_states], 0)  # [L+1, T, HID]
    return HH, rec['hR'], o.logits[0, -1].float().detach().cpu().numpy()

t_cap = time.time()
CAP = {}
for wd, sup in INST_ALL:
    CAP[wd] = capture(TMPL % wd)
w('')
w('--- (P-B) 捕获成本 ---')
w('capture 41 instances in %.2fs (%.4fs each) ; hidden_states shape = %s' %
  (time.time() - t_cap, (time.time() - t_cap) / 41.0, CAP[INST_ALL[0][0]][0].shape))
w('注意：本 Phase 需要【全位置】hidden，故 HH 为 [L+1, T, HID]（Phase 12 只存 [L+1, HID]）')

# ---------------------------------------------------------------- P-C(2) 锚
PAIRS = [p for p in PAIRS_ALL if p[0] in CAP and p[2] in CAP]
H6 = PRIMARY + 1

def est_U(level_idx, insts):
    by = {}
    for wd, sup in insts:
        by.setdefault(sup, []).append(wd)
    avail = [s for s in SUPS if s in by]
    r = max(len(avail) - 1, 1)
    mus = np.stack([np.mean([CAP[wd][0][level_idx][-1] for wd in by[s]], 0) for s in avail], 0).astype(np.float64)
    D = mus - mus.mean(0, keepdims=True)
    _, sv, Vt = np.linalg.svd(D, full_matrices=False)
    return Vt[:r].astype(np.float32), sv[:r], len(avail)

U6, SV6, nA6 = est_U(H6, DISC)
AO = U6
w('')
w('--- (P-C2) 装置锚：U6 与 mean||P_U6(diff6)|| ---')
w('U6 n_classes=%d shape=%s sing=%s' % (nA6, U6.shape, ' '.join('%.2f' % x for x in SV6)))

def proj(vec, Ub):
    return (vec @ Ub.T) @ Ub

n6s = []
for (rw, rs, dw, ds, sw) in PAIRS:
    d6 = CAP[dw][0][H6][-1].astype(np.float32) - CAP[rw][0][H6][-1].astype(np.float32)
    n6s.append(float(np.linalg.norm(proj(d6.astype(np.float64) if d6.dtype == np.float64 else d6, AO))))
N6_MEAN = float(np.mean(n6s))
REF9 = float(E12['mean_n6_ref_from_phase9'])
TOL9 = float(E12['n6_drift_tol'])
w('mean||P_U6(diff6)|| = %.10f ; Phase9 ref = %.10f ; |d| = %.3e ; tol = %.1e ; drift = %s' %
  (N6_MEAN, REF9, abs(N6_MEAN - REF9), TOL9, abs(N6_MEAN - REF9) > TOL9))

# ---------------------------------------------------------------- patch 机制
@torch.no_grad()
def fwd_patch_prefix(text, site, vecs_by_pos):
    """vecs_by_pos: dict pos -> np.array(HID)。把 site（层输出 / R）的【指定位置】替换。
    pos = -1 表示末位（与 Phase 12 完全一致的单点口径）。"""
    ii = torch.tensor([ids_of(text)], device='cuda')
    mod = NORM if site == 'R' else layers[site]
    def hook(m, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        t = t.clone()
        T = t.shape[1]
        for p, v in vecs_by_pos.items():
            pp = T - 1 if p < 0 else p
            t[0, pp, :] = torch.tensor(v, device=t.device, dtype=t.dtype)
        return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t
    h = mod.register_forward_hook(hook)
    try:
        o = model(input_ids=ii)
    finally:
        h.remove()
    return o.logits[0, -1].float().detach().cpu().numpy()

def score_of(v, sup, sid):
    v = v.copy(); v[sid] = -1e9
    own = float(v[SUP_ID[sup]])
    others = [float(v[SUP_ID[x]]) for x in SUPS if x != sup]
    return own - float(np.mean(others))

def rank_of(v, sup, sid):
    v = v.copy(); v[sid] = -1e9
    order = np.argsort(-v)
    return int(np.where(order == SUP_ID[sup])[0][0]) + 1

# 计时：单点 vs 全位点
rw0 = PAIRS[0][0]
T0 = len(ids_of(TMPL % rw0))
h6r = CAP[rw0][0][H6][-1].astype(np.float32)
_ = fwd_patch_prefix(TMPL % rw0, PRIMARY, {-1: h6r})            # warmup
t_a = time.time()
for _ in range(20):
    fwd_patch_prefix(TMPL % rw0, PRIMARY, {-1: h6r})
t_last = (time.time() - t_a) / 20.0
t_a = time.time()
for _ in range(20):
    fwd_patch_prefix(TMPL % rw0, PRIMARY, {p: CAP[rw0][0][H6][p].astype(np.float32) for p in range(T0)})
t_all = (time.time() - t_a) / 20.0
w('')
w('--- (P-B2) 前向成本 ---')
w('T(sample)=%d ; 单点 patch %.4fs/fwd ; 全位点 patch %.4fs/fwd (比值 %.3f)' %
  (T0, t_last, t_all, t_all / max(t_last, 1e-9)))

# 预算估算
N_F1 = len(PROFILE) * 18 * len(PAIRS)      # layer sweep, densified 18-pt grid
N_F2 = maxT * 6 * len(PAIRS)               # position sweep, 6 alphas
N_F3 = 1 * 14 * len(PAIRS)                 # readout R
N_F4 = 4 * 14 * len(CONF)                  # confirmation
N_F5 = 3 * 1 * len(PAIRS)                  # floor
TOT = N_F1 + N_F2 + N_F3 + N_F4 + N_F5
w('')
w('--- (P-B3) 预算估算（全位点 $%.4f/fwd）---' % t_all)
w('  F1 layer sweep   %5d fwd  -> %6.1fs' % (N_F1, N_F1 * t_all))
w('  F2 position sweep%5d fwd  -> %6.1fs' % (N_F2, N_F2 * t_all))
w('  F3 readout R     %5d fwd  -> %6.1fs' % (N_F3, N_F3 * t_all))
w('  F4 confirmation  %5d fwd  -> %6.1fs' % (N_F4, N_F4 * t_all))
w('  F5 floor         %5d fwd  -> %6.1fs' % (N_F5, N_F5 * t_all))
w('  TOTAL            %5d fwd  -> %6.1fs (= %.1f min)' % (TOT, TOT * t_all, TOT * t_all / 60.0))

# ---------------------------------------------------------------- P-C(3) F3 锚
w('')
w('--- (P-C3) α=0 全位点替换 == 未干预前向 ? ---')
devs = []
for (rw, rs, dw, ds, sw) in PAIRS[:5]:
    Tr = len(ids_of(TMPL % rw))
    lg_ref = CAP[rw][2]
    lg0 = fwd_patch_prefix(TMPL % rw, PRIMARY, {p: CAP[rw][0][H6][p].astype(np.float32) for p in range(Tr)})
    devs.append(abs(score_of(lg0, rs, ids_of(rw)[0]) - score_of(lg_ref, rs, ids_of(rw)[0])))
w('  max|dScore(alpha=0, all-positions)| over 5 pairs = %.3e' % max(devs))

w('')
w('--- (P-C4) α=1 全位点替换 == 供体自身前向 ? ---')
devs1 = []
for (rw, rs, dw, ds, sw) in PAIRS[:5]:
    Tr = len(ids_of(TMPL % rw)); Td = len(ids_of(TMPL % dw))
    if Tr != Td:
        w('  skip pair %s->%s (T %d vs %d)' % (rw, dw, Tr, Td)); continue
    lg1 = fwd_patch_prefix(TMPL % rw, PRIMARY, {p: CAP[dw][0][H6][p].astype(np.float32) for p in range(Tr)})
    devs1.append(abs(score_of(lg1, ds, ids_of(dw)[0]) - score_of(CAP[dw][2], ds, ids_of(dw)[0])))
w('  max|dScore(alpha=1, all-positions, donor-score)| over %d pairs = %.3e' % (len(devs1), max(devs1) if devs1 else -1))

w('')
w('--- 结论（供 seal 设计决策） ---')
w('  * 同 T 前提: 所有 41 实例的 T 是否一致 = %s (distinct=%s)' % (len(uniq_T) == 1, uniq_T))
w('  * FULL_SWAP 可逐位继承 Phase 12 = %s' % (FS_REC == float(R12['FULL_SWAP'])))
w('  * n6 锚: |d|=%.3e (tol %.1e) drift=%s' % (abs(N6_MEAN - REF9), TOL9, abs(N6_MEAN - REF9) > TOL9))
w('  * alpha=0 全位点替换还原基线 = %s' % (max(devs) < 1e-2))
w('  * alpha=1 全位点替换 == 供体前向 = %s' % (bool(devs1) and max(devs1) < 1e-2))

io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(lines) + '\n')
print('WROTE', OUT, os.path.getsize(OUT), 'B')
