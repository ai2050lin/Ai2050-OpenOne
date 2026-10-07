# -*- coding: utf-8 -*-
"""
Phase 11 / N2h1-alpha-4 : 自基全剖面 + 噪声带
=========================================================================
预注册：tests/deepseek_temp/Phase11/N2h1a4_design_seal.json（观测前冻结）
执行冻结：tests/deepseek_temp/Phase11/execution_phase11.json

要回答的问题（Phase 10 §8 死线）：
  Phase 10 已定：Q_abs = Q_rel = Q2_accumulate => Q_ROBUST（J(ell) 随深度单调下降，
  Spearman -0.87，全程无断崖）；读数位点 R 线性（排除读数假象）；自基 overlap 0.0298
  说明固定基的深度衰减主要是方向失配 —— 但自基臂只覆盖 4 个位点，且全程无误差带。
  本 Phase：(1) E3 自基臂铺满全部 18 个剖面位点 -> 双基 J(ell) 剖面；
           (2) 逐对 dDonor（n=24）落盘 + 配对 bootstrap -> J(ell) 与 Spearman 的 95% 带；
  判定 Q2（逐层累积）在噪声带下是否仍成立（B1）、是否双基稳健（B2）、判据是否有区分力（F7'）。

用法：python run_phase11.py smoke | python run_phase11.py formal
"""
import os, sys, io, json, time, hashlib
import numpy as np
import torch

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P11T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase11')
EXEC = os.path.join(P11T, 'execution_phase11.json')
SEAL = os.path.join(P11T, 'N2h1a4_design_seal.json')
REPORT = os.path.join(P11T, 'n2h1a4_report_qwen3-4b.txt')
RESULT = os.path.join(P11T, 'result_phase11.json')
SMOKE = os.environ.get('SMOKE', '0') == '1'
if SMOKE:
    _S = os.path.join(P11T, 'smoke')
    os.makedirs(_S, exist_ok=True)
    REPORT = os.path.join(_S, 'n2h1a4_report_qwen3-4b.txt')
    RESULT = os.path.join(_S, 'result_phase11.json')

lines = []


def w(s=''):
    lines.append(str(s)); print(s); sys.stdout.flush()


E = json.load(io.open(EXEC, encoding='utf-8'))
S = json.load(io.open(SEAL, encoding='utf-8'))

# 防御：execution 必需字段自检（首版 gen 曾漏 inherits_* 导致 KeyError）
_REQ = ['inherits_panel_sha256', 'inherits_panel8_sha256', 'inherits_numbers_sha256',
        'phase10_result_for_F9', 'phase10_result_sha256', 'bootstrap',
        'own_basis_sites', 'profile_sites', 'depth_sites', 'arms', 'classifier', 'decision']
_miss = [k for k in _REQ if k not in E]
assert not _miss, 'execution_phase11.json 缺字段: %s（请检查 gen 脚本）' % _miss

MODEL = E['model']
MDIR = os.path.join(ROOT, 'models', 'hf', MODEL)
TMPL = E['template']
SUP_ID = {k: int(v) for k, v in E['sup_id'].items()}
SUPS = E['classes']
PRIMARY = E['primary_layer']
PRE = E['pre_layer']
NH, HD = E['n_heads'], E['head_dim']
RNG = np.random.default_rng(E['seed'])          # E4 地板专用流
RBAR9 = float(E['rbar_ref_from_phase9'])
N6_REF = float(E['mean_n6_ref_from_phase9'])
N6_TOL = float(E['n6_drift_tol'])
FULL_REF = float(E['full_ref_from_phase9'])
D1A_REF = float(E['anchor_ref_d1a_from_phase9'])
ANCHOR_TOL = float(E['anchor_drift_tol'])
PERT_LIM = float(E['off_manifold_pert_rel'])
CL = E['classifier']
DEC = E['decision']
BOOT = E['bootstrap']
DEPTH = list(E['depth_sites'])
PROFILE = list(E['profile_sites'])          # = [6] + depth_sites（18 个位点）
R_SITE = E['readout_site']
OB_SITES = list(E['own_basis_sites'])       # 本 Phase = 全部 18 个剖面位点
FL_SITES = list(E['floor_sites'])
CF_SITES = list(E['conf_sites'])
F3_SITES = list(E['f3_sites'])
E1_SITES = PROFILE + [R_SITE]


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


w('=== Phase 11 / N2h1-alpha-4 : 自基全剖面 + 噪声带 ===')
w('smoke=%s ; time %s' % (SMOKE, time.strftime('%Y-%m-%d %H:%M:%S')))
w('seal sha8 %s ; exec sha8 %s' % (sha(SEAL)[:8], sha(EXEC)[:8]))
w('exec inherits panel(phase10) sha256 %s' % E['inherits_panel_sha256'][:16])
w('exec inherits panel(phase8)  sha256 %s' % E['inherits_panel8_sha256'][:16])
w('exec inherits numbers(phase10 result) sha256 %s' % E['inherits_numbers_sha256'][:16])
w('own_basis_sites(%d) = %s' % (len(OB_SITES), OB_SITES))
w('config_sha256_match %s (expect %s)' % (
    sha(os.path.join(MDIR, 'config.json')) == E['config_sha256'], E['config_sha256'][:12]))

# ---- F9 参照：Phase 10 的 E1 曲线 ----
R10P = os.path.join(ROOT, E['phase10_result_for_F9'])
assert sha(R10P) == E['phase10_result_sha256'], 'F9 前置：result_phase10.json 已漂移'
R10 = json.load(io.open(R10P, encoding='utf-8'))

from transformers import AutoTokenizer, AutoModelForCausalLM
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MDIR, dtype=torch.bfloat16, trust_remote_code=True,
                                            attn_implementation='eager').to('cuda').eval()
_core = getattr(model.model, 'language_model', model.model)
layers = _core.layers
L = len(layers)
head_lm = model.lm_head
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
assert NORM is not None, '找不到最终 norm 模块，无法建 R 位点'


def ids_of(s):
    return tok.encode(s, add_special_tokens=False)


def attn_out_proj(layer):
    a = layer.self_attn
    for nm in ['o_proj', 'dense', 'out_proj']:
        if hasattr(a, nm):
            return getattr(a, nm)
    raise RuntimeError('no attn out proj')


ATTN = [attn_out_proj(layers[i]) for i in range(L)]
OIN = ATTN[PRIMARY].in_features

# ---------------- 0. drift 断言（F4）----------------
drift = []
if L != E['expected_cfg']['num_hidden_layers']: drift.append('layers')
if HID != E['expected_cfg']['hidden_size']: drift.append('hidden')
if CFG.num_attention_heads != NH: drift.append('n_heads')
if bool(CFG.tie_word_embeddings) != bool(E['expected_cfg']['tie_word_embeddings']): drift.append('tie')
if OIN != E['o_proj_in_features']: drift.append('o_proj_in %d' % OIN)
w('drift(F4): %s' % (drift if drift else 'NONE'))
assert OIN == NH * HD, 'F4 失败：o_proj 输入维 %d != %d' % (OIN, NH * HD)
if drift:
    w('!! DRIFT 非空，按预注册 F4 停止'); sys.exit(2)
assert max(DEPTH) < L - 1, 'depth_sites 超出层数'

PAIRS_ALL = [tuple(p) for p in E['pairs_all']]
DISC = [tuple(x) for x in E['discovery']]
CONF = [tuple(x) for x in E['confirmation']]
INST_ALL = [tuple(x) for x in E['instances_all']]

if SMOKE:
    DISC = [DISC[i] for i in (0, 4, 8, 12, 16, 20)]
    CONF = []
    PROFILE = [6, 7, 8]
    OB_SITES = list(PROFILE)
    E1_SITES = PROFILE + [R_SITE]
    CF_SITES = []
    _need = set()
    for p in PAIRS_ALL:
        if p[0] in [x[0] for x in DISC]:
            _need.add(p[0]); _need.add(p[2])
    INST_ALL = [(a, b) for (a, b) in INST_ALL if a in _need]
    w('SMOKE panel: discovery=%d ; captured=%s ; profile=%s' %
      (len(DISC), [x[0] for x in INST_ALL], PROFILE))


def _keep_sites(lst):
    return [s for s in lst if s == R_SITE or s in PROFILE]


F3_SITES = _keep_sites(F3_SITES)
FL_SITES = _keep_sites(FL_SITES)
CF_SITES = _keep_sites(CF_SITES)


def grid_of(arm, n_smoke=3):
    g = E['arms'][arm]['alpha_grid']
    if isinstance(g, str):
        return g
    if not SMOKE:
        return list(g)
    sub = list(g[:n_smoke])
    if 1.0 in g and 1.0 not in sub:
        sub.append(1.0)
    return sub


w('')
w('model=%s L=%d hid=%d heads=%d head_dim=%d o_proj_in=%d tie=%s' %
  (MODEL, L, HID, NH, HD, OIN, CFG.tie_word_embeddings))
w('template=%r ; seed=%d ; norm module=%s' % (TMPL, E['seed'], type(NORM).__name__))
w('E1 sites (%d) = %s' % (len(E1_SITES), E1_SITES))
w('panel: discovery=%d confirmation=%d (all=%d)' % (len(DISC), len(CONF), len(INST_ALL)))
sys.stdout.flush()

# ---------------- 1. 采集（全 hidden states + norm 输出）----------------
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
    HH = np.stack([x[0, -1].float().detach().cpu().numpy() for x in o.hidden_states], 0)
    return HH, rec['hR'], o.logits[0, -1].float().detach().cpu().numpy()


t_cap = time.time()
CAP = {}
for wd, sup in INST_ALL:
    CAP[wd] = capture(TMPL % wd)
w('capture done %d instances in %.1fs (hidden_levels=%d)' % (len(CAP), time.time() - t_cap, CAP[INST_ALL[0][0]][0].shape[0]))
if SMOKE:
    wd0 = INST_ALL[0][0]
    HH, hR, lg = CAP[wd0]
    w('SMOKE assert: HH%s hR%s logits%s nan=%s' % (HH.shape, hR.shape, lg.shape, bool(np.isnan(HH).any())))
    assert HH.shape[1] == HID and hR.shape[0] == HID and not np.isnan(HH).any()
PAIRS = [p for p in PAIRS_ALL if p[0] in CAP and p[2] in CAP]
w('usable pairs (recipient & donor captured) = %d / %d' % (len(PAIRS), len(PAIRS_ALL)))

H6 = PRIMARY + 1   # hidden_states[7] = L6 输出
sys.stdout.flush()


# ---------------- 2. 类子空间 U6（discovery only）----------------
def est_U(level_idx, insts):
    by = {}
    for wd, sup in insts:
        if wd in CAP:
            by.setdefault(sup, []).append(wd)
    avail = [s for s in SUPS if s in by]
    r = max(len(avail) - 1, 1)
    mus = np.stack([np.mean([CAP[wd][0][level_idx] for wd in by[s]], 0) for s in avail], 0).astype(np.float64)
    D = mus - mus.mean(0, keepdims=True)
    _, sv, Vt = np.linalg.svd(D, full_matrices=False)
    return Vt[:r].astype(np.float32), sv[:r], len(avail)


U6, SV6, nA6 = est_U(H6, DISC)
AO = U6
w('U6 estimated on discovery: n_classes=%d ; shape %s ; sing = %s' %
  (nA6, U6.shape, ' '.join('%.2f' % x for x in SV6)))
sys.stdout.flush()


def proj(vec, Ub):
    return (vec @ Ub.T) @ Ub


# ---------------- 3. base 分数 ----------------
def score_of(v, sup, sid):
    v = v.copy(); v[sid] = -1e9
    own = float(v[SUP_ID[sup]])
    others = [float(v[SUP_ID[x]]) for x in SUPS if x != sup]
    return own - float(np.mean(others))


def rank_of(v, sup, sid):
    v = v.copy(); v[sid] = -1e9
    order = np.argsort(-v)
    return int(np.where(order == SUP_ID[sup])[0][0]) + 1


BASE = {}
for (rw, rs, dw, ds, sw) in PAIRS:
    lg = CAP[rw][2]
    BASE[rw] = dict(sr0=score_of(lg, rs, ids_of(rw)[0]),
                    sd0=score_of(lg, ds, ids_of(dw)[0]),
                    rd0=rank_of(lg, ds, ids_of(dw)[0]))
bad_base = [rw for rw, b in BASE.items() if not (b['sr0'] > 0)]
w('base: n=%d ; 受体类分数 mean=%+.3f ; 供体类分数 mean=%+.3f ; 供体类已 rank1 比例=%.3f ; base<=0 = %s (F2)' %
  (len(BASE), np.mean([b['sr0'] for b in BASE.values()]), np.mean([b['sd0'] for b in BASE.values()]),
   float(np.mean([1.0 if BASE[rw]['rd0'] == 1 else 0.0 for rw in BASE])), bad_base if bad_base else 'NONE'))
assert not bad_base, 'F2 失败'
sys.stdout.flush()

# ---------------- 4. 每对的注入向量与各位点基准态 ----------------
H_SITES = sorted(set(PROFILE + [PRIMARY]))   # 含 L6 输出
VEC = {}
for (rw, rs, dw, ds, sw) in PAIRS:
    h6r, h6d = CAP[rw][0][H6].astype(np.float32), CAP[dw][0][H6].astype(np.float32)
    d6 = h6d - h6r
    u6 = proj(d6, AO)
    n6 = float(np.linalg.norm(u6))
    d = dict(u6=u6, n6=n6, unit6=(u6 / max(n6, 1e-9)),
             hR=CAP[rw][1].astype(np.float32), nhR=float(np.linalg.norm(CAP[rw][1])),
             h_ell={}, nh_ell={}, d_ell={})
    for s in H_SITES:
        hr = CAP[rw][0][s + 1].astype(np.float32)
        hd = CAP[dw][0][s + 1].astype(np.float32)
        d['h_ell'][s] = hr
        d['nh_ell'][s] = float(np.linalg.norm(hr))
        d['d_ell'][s] = hd - hr
    VEC[rw] = d

N6_MEAN = float(np.mean([VEC[rw]['n6'] for rw in VEC]))
N6_DRIFT = abs(N6_MEAN - N6_REF) > N6_TOL
_RB = np.array([[VEC[rw]['n6'] / max(VEC[rw]['nh_ell'][s], 1e-9) for s in H_SITES] for rw in VEC])
RBAR_ELL = {s: float(np.mean(_RB[:, i])) for i, s in enumerate(H_SITES)}
R_R = float(np.mean([VEC[rw]['n6'] / max(VEC[rw]['nhR'], 1e-9) for rw in VEC]))
w('')
w('--- 剂量尺度 ---')
w('mean||P_U6(diff6)|| = %.4f (Phase 9/10 参照 %.4f, drift=%s)' % (N6_MEAN, N6_REF, N6_DRIFT))
w('相对剂量 r_ell = mean||u6||/||h_ell_recip|| : %s' %
  '  '.join('L%d:%.4f' % (s, RBAR_ELL[s]) for s in H_SITES))
w('  [口径声明] r_ell 与 Phase 9 的 rbar=%.5f 不是同一个量，不作数值比对。' % RBAR9)
w('  r_R = %.5f' % R_R)
sys.stdout.flush()


# ---------------- 5. 前向工具 ----------------
@torch.no_grad()
def fwd_patch(text, site, vec):
    ii = torch.tensor([ids_of(text)], device='cuda')
    mod = NORM if site == R_SITE else layers[site]

    def hook(m, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        t = t.clone(); t[0, -1, :] = vec.to(t.dtype)
        return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t

    h = mod.register_forward_hook(hook)
    try:
        o = model(input_ids=ii)
    finally:
        h.remove()
    return o.logits[0, -1].float().detach().cpu().numpy()


def base_state(V, site):
    return (V['hR'], V['nhR']) if site == R_SITE else (V['h_ell'][site], V['nh_ell'][site])


# ---------------- 6. F3：alpha=0 恒等自检（含 R 位点）----------------
w('')
w('--- F3 自检：alpha=0 注入必须与未干预前向同分（含 R 位点 norm hook）---')
f3 = {}
for site in F3_SITES:
    devs = []
    for (rw, rs, dw, ds, sw) in PAIRS[:3]:
        h0, _ = base_state(VEC[rw], site)
        lg0 = fwd_patch(TMPL % rw, site, torch.tensor(h0, device='cuda'))
        devs.append(abs(score_of(lg0, rs, ids_of(rw)[0]) - BASE[rw]['sr0']))
    f3[str(site)] = float(max(devs))
w('  max|dScore(alpha=0)| : %s' % '  '.join('%s=%.3e' % (k, v) for k, v in f3.items()))
assert all(v < 1e-2 for v in f3.values()), 'F3 失败：某位点钩子未还原基线'
sys.stdout.flush()


# ---------------- 7. 剂量执行器（本 Phase 一律逐对落盘）----------------
def _acc(lg, B, rw, rs, dw, ds):
    sid_r, sid_d = ids_of(rw)[0], ids_of(dw)[0]
    return (score_of(lg, ds, sid_d) - B['sd0'],
            score_of(lg, rs, sid_r) - B['sr0'],
            1 if rank_of(lg, ds, sid_d) == 1 else 0)


def dose_abs(site, alpha_list, pairs, per_pair=True):
    out = []
    for a in alpha_list:
        dd = dr = 0.0; r1 = 0; n = 0; perts = []; per = []; order = []
        for (rw, rs, dw, ds, sw) in pairs:
            if rw not in VEC:
                continue
            V, B = VEC[rw], BASE[rw]
            h0, nh0 = base_state(V, site)
            lg = fwd_patch(TMPL % rw, site, torch.tensor(h0 + a * V['u6'], device='cuda'))
            x1, x2, x3 = _acc(lg, B, rw, rs, dw, ds)
            dd += x1; dr += x2; r1 += x3; n += 1
            perts.append(a * V['n6'] / max(nh0, 1e-9))
            per.append(float(x1)); order.append(rw)
        row = dict(alpha=float(a), dDonor=dd / max(n, 1), dRecip=dr / max(n, 1),
                   rank1=r1 / max(n, 1), n=n, pert_rel=float(np.mean(perts)))
        if per_pair:
            row['per_pair'] = per; row['order'] = order
        out.append(row)
    return out


def dose_rel(site, arel_list, pairs, per_pair=True):
    out = []
    for a in arel_list:
        dd = 0.0; r1 = 0; n = 0; per = []; order = []
        for (rw, rs, dw, ds, sw) in pairs:
            if rw not in VEC:
                continue
            V, B = VEC[rw], BASE[rw]
            h0, nh0 = base_state(V, site)
            lg = fwd_patch(TMPL % rw, site, torch.tensor(h0 + a * nh0 * V['unit6'], device='cuda'))
            x1, x2, x3 = _acc(lg, B, rw, rs, dw, ds)
            dd += x1; r1 += x3; n += 1
            per.append(float(x1)); order.append(rw)
        row = dict(alpha_rel=float(a), dDonor=dd / max(n, 1), rank1=r1 / max(n, 1), n=n)
        if per_pair:
            row['per_pair'] = per; row['order'] = order
        out.append(row)
    return out


DISC_P = [p for p in PAIRS if p[0] in [x[0] for x in DISC]]
CONF_P = [p for p in PAIRS if p[0] in [x[0] for x in CONF]]

# ---------------- 8. 锚点 ----------------
w('')
w('--- E0 锚点：S_L6out, alpha=1（内建跨 Phase 复现点）---')
_gl = grid_of('E0_anchor_L6')
E0 = dose_abs(PRIMARY, _gl, DISC_P, per_pair=False)[0]
w('  dDonor=%+.15f ; 参照 %.15f ; 逐位相等=%s' %
  (E0['dDonor'], FULL_REF, (E0['dDonor'] == FULL_REF)))
if not SMOKE:
    assert E0['dDonor'] == FULL_REF, 'F6 失败：E0 未逐位复现 Phase 9 的 full'
FULL = float(E0['dDonor'])
w('full_L6 := %.15f' % FULL)

w('')
w('--- E0b 读出口径锚点：S_L6out, alpha=rbar9=%.5f（应对齐 Phase 9 D1a=%+.4f）---' % (RBAR9, D1A_REF))
E0b = dose_abs(PRIMARY, [RBAR9], DISC_P, per_pair=False)[0]
AD = abs(E0b['dDonor'] - D1A_REF) > ANCHOR_TOL
w('  dDonor=%+8.4f ; ref %+.4f ; |d|=%.4f ; drift=%s' % (E0b['dDonor'], D1A_REF, abs(E0b['dDonor'] - D1A_REF), AD))
sys.stdout.flush()

# ---------------- 9. E1 绝对剂量深度剖面（逐对落盘）----------------
w('')
w('--- E1 绝对剂量深度剖面（h_ell + alpha * P_U6(diff6)，discovery n=%d，逐对落盘）---' % len(DISC_P))
G1 = grid_of('E1_depth_abs')
E1 = {}
E1P = {}
for s in E1_SITES:
    r = dose_abs(s, G1, DISC_P, per_pair=True)
    E1[str(s)] = r
    E1P[str(s)] = [x.get('per_pair', []) for x in r]
    w('  L%-3s %s' % (s, '  '.join('a=%.3f dD=%+8.3f pr=%.2f' % (x['alpha'], x['dDonor'], x['pert_rel']) for x in r)))
sys.stdout.flush()

# ---------------- 10. E1b 相对剂量深度剖面（逐对落盘）----------------
w('')
w('--- E1b 相对剂量深度剖面（h_ell + a_rel*||h_ell||*unit(u6)）---')
G2 = grid_of('E1b_depth_rel', 2)
E1b = {}
E1bP = {}
for s in E1_SITES:
    r = dose_rel(s, G2, DISC_P, per_pair=True)
    E1b[str(s)] = r
    E1bP[str(s)] = [x.get('per_pair', []) for x in r]
    w('  L%-3s %s' % (s, '  '.join('r=%.2f dD=%+8.3f' % (x['alpha_rel'], x['dDonor']) for x in r)))
sys.stdout.flush()

# ---------------- 10b. E6 读数位点扩展网格 ----------------
w('')
w('--- E6 读数位点 R 扩展网格 ---')
G6 = grid_of('E6_readout_grid')
E6 = dose_abs(R_SITE, G6, DISC_P, per_pair=False)
for r in E6:
    w('  R  a=%7.3f  dD=%+8.3f  rank1=%.3f  pert_rel=%.3f' % (r['alpha'], r['dDonor'], r['rank1'], r['pert_rel']))
sys.stdout.flush()

# ---------------- 11. E3 自基全剖面（逐对落盘）----------------
w('')
w('--- E3 自基全剖面（h_ell + alpha * P_U{ell}(diff_ell)，%d 个位点）---' % len(OB_SITES))
G3 = grid_of('E3_own_basis')
E3 = {}
E3P = {}
for s in OB_SITES:
    Uell, SVell, nA = est_U(s + 1, DISC)
    Mc = Uell @ AO.T
    svc = np.linalg.svd(Mc, compute_uv=False)
    ov = float(np.sum(svc ** 2) / AO.shape[0])
    out = []
    per_pair_alpha = []
    for a in G3:
        dd = 0.0; n = 0; per = []; order = []
        for (rw, rs, dw, ds, sw) in DISC_P:
            if rw not in VEC:
                continue
            V, B = VEC[rw], BASE[rw]
            u_own = proj(V['d_ell'][s], Uell)
            lg = fwd_patch(TMPL % rw, s, torch.tensor(V['h_ell'][s] + a * u_own, device='cuda'))
            x1 = score_of(lg, ds, ids_of(dw)[0]) - B['sd0']
            dd += x1; n += 1; per.append(float(x1)); order.append(rw)
        out.append(dict(alpha=float(a), dDonor=dd / max(n, 1), n=n, order=order))
        per_pair_alpha.append(per)
    E3[str(s)] = dict(rows=out, principal_cos=[float(x) for x in svc], overlap=ov,
                      n_classes=nA, sing=[float(x) for x in SVell])
    E3P[str(s)] = per_pair_alpha
    w('  L%-3s overlap(U%d,U6)=%.4f  %s' % (s, s, ov,
                                            '  '.join('a=%.2f dD=%+8.3f' % (x['alpha'], x['dDonor']) for x in out)))
sys.stdout.flush()

# ---------------- 12. E4 地板 ----------------
w('')
w('--- E4 地板（U6 内随机 5 维，范数匹配，2 draws）---')
G4 = grid_of('E4_floor')
_drw = E['arms']['E4_floor']['draws']
E4 = []
for s in FL_SITES:
    dd, n = 0.0, 0
    for (rw, rs, dw, ds, sw) in DISC_P:
        if rw not in VEC:
            continue
        V, B = VEC[rw], BASE[rw]
        h0, _nh = base_state(V, s)
        for _ in range(_drw):
            z = RNG.standard_normal(AO.shape[0]).astype(np.float32)
            v = z @ AO
            v = v / max(np.linalg.norm(v), 1e-9) * V['n6']
            lg = fwd_patch(TMPL % rw, s, torch.tensor(h0 + v, device='cuda'))
            dd += score_of(lg, ds, ids_of(dw)[0]) - B['sd0']; n += 1
    E4.append(dict(site=s, alpha=1.0, dDonor=dd / max(n, 1), n=n))
    w('  L%-3s dDonor=%+8.4f (n=%d)' % (s, E4[-1]['dDonor'], n))
sys.stdout.flush()


# ---------------- 13. 参数化曲线分类器（与 Phase 10 逐字相同）----------------
def _rank(a):
    a = np.asarray(a, float)
    order = np.argsort(a)
    r = np.empty(len(a), float)
    r[order] = np.arange(len(a), dtype=float)
    return r


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 1e-12 else 0.0


def curve_stats(xs, ys, tag=''):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    m = xs >= 0.01
    xs2, ys2 = xs[m], ys[m]
    det = dict(x=xs.tolist(), y=ys.tolist(), n=len(xs))
    if len(xs2) < 3:
        det.update(jump_ratio=None, slopes=[], x_star=None, k_log=None, y_sat=float(np.max(ys)), R2_log=None,
                   R2_lin=None, gamma=None, cls='UNCLASSIFIED', reason='too_few_points')
        return det
    s = np.diff(ys2) / np.diff(xs2)
    k_i = int(np.argmax(s)); s_med = float(np.median(np.delete(s, k_i))) if len(s) > 1 else 0.0
    J = float(s[k_i] / s_med) if s_med > 1e-12 else float('inf')
    det['slopes'] = s.tolist(); det['jump_ratio'] = J; det['argmax_slope_x'] = float(xs2[k_i])
    mf = xs >= 0.10
    xf, yf = xs[mf], ys[mf]
    R2l = None
    if len(xf) >= 3:
        b1, b0 = np.polyfit(xf, yf, 1)
        R2l = float(1.0 - np.sum((yf - (b0 + b1 * xf)) ** 2) / max(np.sum((yf - yf.mean()) ** 2), 1e-12))
    gam = None
    if len(xf) >= 3 and np.all(yf > 0):
        g, _lc = np.polyfit(np.log(xf), np.log(yf), 1)
        gam = float(g)
    R2g, k_log, x_star = None, None, None
    A = float(np.max(ys))
    if A > 1e-9:
        kk = np.arange(CL['logistic_k_min'], CL['logistic_k_max'] + 1e-9, CL['logistic_k_step'])
        x0 = np.arange(xs.min(), xs.max() + 1e-9, CL['logistic_x0_step'])
        P = A / (1.0 + np.exp(-(kk[:, None, None] * (xs[None, None, :] - x0[None, :, None]))))
        SSE = ((P - ys[None, None, :]) ** 2).sum(axis=2)
        ij = np.unravel_index(int(np.argmin(SSE)), SSE.shape)
        sse = float(SSE[ij])
        SSt = float(np.sum((ys - ys.mean()) ** 2))
        R2g = float(1.0 - sse / max(SSt, 1e-12))
        k_log = float(kk[ij[0]]); x_star = float(x0[ij[1]])
    det.update(R2_log=R2g, k_log=k_log, x_star=x_star, R2_lin=R2l, gamma=gam,
               y_sat=float(np.max(np.abs(ys))), y_sat_signed=float(np.max(ys)))
    if det['y_sat'] < CL['UNREACH_y']:
        cls = 'UNREACH'
    elif R2g is not None and R2g >= CL['S_STRONG']['R2_log'] and J >= CL['S_STRONG']['J'] and k_log >= CL['S_STRONG']['k_log']:
        cls = 'S_STRONG'
    elif R2g is not None and R2g >= CL['S_WEAK']['R2_log'] and J >= CL['S_WEAK']['J']:
        cls = 'S_WEAK'
    elif R2g is not None and R2g >= CL['GRADUAL']['R2_log']:
        cls = 'GRADUAL'
    elif R2l is not None and R2l >= CL['LINEAR']['R2_lin'] and (R2g is None or R2g < CL['LINEAR']['R2_log_max']):
        cls = 'LINEAR'
    else:
        cls = 'UNCLASSIFIED'
    det['cls'] = cls
    return det


prof_abs, prof_rel = {}, {}
for s in E1_SITES:
    rows = E1[str(s)]
    prof_abs[str(s)] = curve_stats([r['alpha'] for r in rows], [r['dDonor'] / FULL for r in rows], 'abs')
    rr = E1b[str(s)]
    prof_rel[str(s)] = curve_stats([r['alpha_rel'] for r in rr], [r['dDonor'] / FULL for r in rr], 'rel')
prof_R_ext = curve_stats([r['alpha'] for r in E6], [r['dDonor'] / FULL for r in E6], 'Rext')
XSTAR_REL = {}
for s in E1_SITES:
    xs = prof_abs[str(s)]['x_star']
    rl = R_R if s == R_SITE else RBAR_ELL.get(s)
    XSTAR_REL[str(s)] = (xs * rl) if (xs is not None and rl) else None


# ---------------- 13b. bootstrap 与置换零假设 ----------------
def J_only(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    m = xs >= 0.01
    xs2, ys2 = xs[m], ys[m]
    if len(xs2) < 3:
        return np.nan
    s = np.diff(ys2) / np.diff(xs2)
    k_i = int(np.argmax(s))
    rest = np.delete(s, k_i)
    s_med = float(np.median(rest)) if len(s) > 1 else 0.0
    return float(s[k_i] / s_med) if s_med > 1e-12 else float('inf')


def J_batch(xs, Y):
    """Y: (n_sites, n_alpha) -> J 数组 (n_sites,)"""
    xs = np.asarray(xs, float)
    m = xs >= 0.01
    x2 = xs[m]; Y2 = np.asarray(Y, float)[:, m]
    ns = Y2.shape[0]
    if len(x2) < 3:
        return np.full(ns, np.nan)
    S = np.diff(Y2, axis=1) / np.diff(x2)
    ai = np.argmax(S, axis=1)
    out = np.empty(ns)
    for i in range(ns):
        rest = np.delete(S[i], ai[i])
        med = float(np.median(rest)) if len(S[i]) > 1 else 0.0
        out[i] = S[i, ai[i]] / med if med > 1e-12 else np.inf
    return out


BS = BOOT['B'] if not SMOKE else 200
BP = BOOT['B_perm'] if not SMOKE else 200
bseed = int(BOOT['seed'])
BRNG = np.random.default_rng(bseed)          # bootstrap 专用独立流


def boot_spearman(pair_mats, xs, sites, B, rng):
    """pair_mats: (n_sites, n_alpha, n_pairs)（原始 dDonor）；返回 B 个 Spearman 与 B 个 J 矩阵"""
    n_sites, n_alpha, n_pairs = pair_mats.shape
    site_idx = np.array(sites, dtype=float)
    sp = np.empty(B)
    Jmat = np.empty((B, n_sites))
    nan_ct = 0
    for b in range(B):
        idx = rng.integers(0, n_pairs, n_pairs)
        Y = pair_mats[:, :, idx].mean(axis=2) / FULL       # (n_sites, n_alpha)
        Jv = J_batch(xs, Y)
        Jmat[b] = Jv
        ok = np.isfinite(Jv)
        if ok.sum() < 4:
            sp[b] = np.nan; nan_ct += 1; continue
        sp[b] = spearman(Jv[ok], site_idx[ok])
    return sp, Jmat, nan_ct


def boot_stats(sp):
    sp2 = sp[np.isfinite(sp)]
    if len(sp2) < 10:
        return dict(lo=None, hi=None, med=None, n_ok=int(len(sp2)))
    lo, hi = np.percentile(sp2, [2.5, 97.5])
    return dict(lo=float(lo), hi=float(hi), med=float(np.median(sp2)), n_ok=int(len(sp2)))


# --- 位点集：J 有效的 E1 剖面位点（对齐 Phase 10 的 Lq）---
Lq = [s for s in PROFILE if prof_abs[str(s)]['jump_ratio'] is not None and np.isfinite(prof_abs[str(s)]['jump_ratio'])]
Lr = [s for s in PROFILE if prof_rel[str(s)]['jump_ratio'] is not None and np.isfinite(prof_rel[str(s)]['jump_ratio'])]
J_abs_hat = np.array([prof_abs[str(s)]['jump_ratio'] for s in Lq])
J_rel_hat = np.array([prof_rel[str(s)]['jump_ratio'] for s in Lr])
rho_abs_hat = spearman(J_abs_hat, np.array(Lq, float)) if len(Lq) >= 4 else None
rho_rel_hat = spearman(J_rel_hat, np.array(Lr, float)) if len(Lr) >= 4 else None

_skip_boot = SMOKE or len(Lq) < 4
if not _skip_boot:
    PM_abs = np.stack([np.array(E1P[str(s)], dtype=float) for s in Lq], 0)      # (nS, nA, nP)
    PM_rel = np.stack([np.array(E1bP[str(s)], dtype=float) for s in Lr], 0)
    xs_abs = np.array([r['alpha'] for r in E1[str(Lq[0])]], float)
    xs_rel = np.array([r['alpha_rel'] for r in E1b[str(Lr[0])]], float)
    sp_abs, Jm_abs, nc_abs = boot_spearman(PM_abs, xs_abs, Lq, BS, BRNG)
    sp_rel, Jm_rel, nc_rel = boot_spearman(PM_rel, xs_rel, Lr, BS, BRNG)
    ST_abs, ST_rel = boot_stats(sp_abs), boot_stats(sp_rel)
    # 位点 J 的 95% 区间（半宽）
    Jci = {}
    for i, s in enumerate(Lq):
        col = Jm_abs[:, i]; col = col[np.isfinite(col)]
        if len(col) >= 10:
            q25, q975 = np.percentile(col, [2.5, 97.5])
            Jci[str(s)] = dict(lo=float(q25), hi=float(q975), hat=float(J_abs_hat[i]),
                               half=float((q975 - q25) / 2.0))
        else:
            Jci[str(s)] = dict(lo=None, hi=None, hat=float(J_abs_hat[i]), half=None)
    # 置换零假设
    perm = np.empty(BP)
    order_idx = np.array(Lq, float)
    for b in range(BP):
        Js = BRNG.permutation(J_abs_hat)
        perm[b] = spearman(Js, order_idx)
    pr = perm[np.isfinite(perm)]
    PLO, PHI = (float(np.percentile(pr, 2.5)), float(np.percentile(pr, 97.5))) if len(pr) >= 10 else (None, None)
else:
    sp_abs = sp_rel = np.array([]); Jm_abs = Jm_rel = None
    ST_abs = ST_rel = dict(lo=None, hi=None, med=None, n_ok=0)
    Jci = {}; PLO = PHI = None; nc_abs = nc_rel = 0
    perm = np.array([])

# --- E3 自基全剖面的 J 与带 ---
J_own = {}
for s in OB_SITES:
    rows = E3[str(s)]['rows']
    J_own[str(s)] = J_only([r['alpha'] for r in rows], [r['dDonor'] / FULL for r in rows])
Lo = [s for s in OB_SITES if np.isfinite(J_own[str(s)])]
J_own_hat = np.array([J_own[str(s)] for s in Lo]) if Lo else np.array([])
spread_own = (float(J_own_hat.max() / max(J_own_hat.min(), 1e-9)) if len(J_own_hat) else None)
rho_own_hat = spearman(J_own_hat, np.array(Lo, float)) if len(Lo) >= 4 else None

if not _skip_boot and len(Lo) >= 4:
    PM_own = np.stack([np.array(E3P[str(s)], dtype=float) for s in Lo], 0)
    xs_own = np.array([r['alpha'] for r in E3[str(Lo[0])]['rows']], float)
    sp_own, Jm_own, _nco = boot_spearman(PM_own, xs_own, Lo, BS, BRNG)
    ST_own = boot_stats(sp_own)
    spread_ow_ci = None
    if Jm_own is not None:
        sp_ci = []
        for b in range(Jm_own.shape[0]):
            row = Jm_own[b][np.isfinite(Jm_own[b])]
            if len(row) >= 4:
                sp_ci.append(row.max() / max(row.min(), 1e-9))
        if len(sp_ci) >= 10:
            _a, _b = np.percentile(sp_ci, [2.5, 97.5])
            spread_ow_ci = dict(lo=float(_a), hi=float(_b))
else:
    ST_own = dict(lo=None, hi=None, med=None, n_ok=0); spread_ow_ci = None

# --- E5 确认集（同时给 Spearman 带 B4）---
w('')
conf_out = {}
if CF_SITES and CONF_P:
    w('--- E5 确认集验带（n=%d，位点 %s）---' % (len(CONF_P), CF_SITES))
    conf_out['sites'] = {}
    confP = {}
    for s in CF_SITES:
        rows = dose_abs(s, G1, CONF_P, per_pair=True)
        d = curve_stats([r['alpha'] for r in rows], [r['dDonor'] / FULL for r in rows], 'conf')
        confP[str(s)] = [x.get('per_pair', []) for x in rows]
        conf_out['sites'][str(s)] = dict(rows=rows, cls=d['cls'], J=d['jump_ratio'], x_star=d['x_star'],
                                         y_sat=d['y_sat'], R2_log=d['R2_log'], same_cls=bool(d['cls'] == prof_abs[str(s)]['cls']))
        w('  L%-3s %s  J=%s x*=%s y_sat=%.3f 与发现集同判=%s' %
          (s, d['cls'], ('%.2f' % d['jump_ratio']) if d['jump_ratio'] is not None else 'n/a',
           ('%.3f' % d['x_star']) if d['x_star'] is not None else 'n/a', d['y_sat'],
           conf_out['sites'][str(s)]['same_cls']))
    conf_out['same_cls_frac'] = float(np.mean([1.0 if v['same_cls'] else 0.0 for v in conf_out['sites'].values()]))
    # B4：确认集上的 Spearman 带（n=17，更宽）
    Lc = [s for s in CF_SITES if conf_out['sites'][str(s)]['J'] is not None and np.isfinite(conf_out['sites'][str(s)]['J'])]
    if not SMOKE and len(Lc) >= 4:
        PM_c = np.stack([np.array(confP[str(s)], dtype=float) for s in Lc], 0)
        xs_c = np.array([r['alpha'] for r in conf_out['sites'][str(Lc[0])]['rows']], float)
        sp_c, _Jc, _nc = boot_spearman(PM_c, xs_c, Lc, BS, BRNG)
        conf_out['spearman_boot'] = dict(hat=spearman(np.array([conf_out['sites'][str(s)]['J'] for s in Lc]),
                                                     np.array(Lc, float)), **boot_stats(sp_c))
    else:
        conf_out['spearman_boot'] = None
else:
    w('  (确认集在冒烟下跳过)')
sys.stdout.flush()

# ---------------- 14. 判据（P/Q 复现 + B 族）----------------
Lp = [s for s in PROFILE if prof_abs[str(s)]['cls'] != 'UNREACH']
_gr = {}
# P4
win = [s for s in Lp if DEC['P4']['window'][0] <= s <= DEC['P4']['window'][1]]
Ja = prof_abs['7']['jump_ratio'] if '7' in prof_abs else None
p4 = bool(win and len(win) >= 2 and Ja is not None and Ja >= DEC['P4']['J_anchor'] and
          all(prof_abs[str(s)]['jump_ratio'] is not None and prof_abs[str(s)]['jump_ratio'] >= DEC['P4']['J_anchor'] for s in win))
_gr['P4_once_formed'] = p4; _gr['P4_window'] = win
# P1
p1 = False; ell0 = None
for i, s in enumerate(Lp):
    J = prof_abs[str(s)]['jump_ratio']
    if J is None or J < DEC['P1']['J']:
        continue
    if any((prof_abs[str(t)]['jump_ratio'] or 0) >= DEC['P1']['J'] for t in Lp[:i]):
        continue
    deep = Lp[i + 1:]
    if any((prof_abs[str(t)]['jump_ratio'] or 0) > DEC['P1']['post_slack'] * J for t in deep):
        continue
    xs0 = prof_abs[str(s)]['x_star']
    xs_all = [prof_abs[str(t)]['x_star'] for t in deep]
    if xs0 and all(x is None or abs(x - xs0) / xs0 <= DEC['P1']['x_star_tol'] for x in xs_all):
        p1 = True; ell0 = s; break
_gr['P1_single_source'] = p1; _gr['P1_ell0'] = ell0
# P2
p2 = bool(rho_abs_hat is not None and rho_abs_hat >= DEC['P2']['spearman_min'] and
          len(Lq) >= 2 and prof_abs[str(Lq[-1])]['jump_ratio'] >= DEC['P2']['ratio_min'] * max(prof_abs[str(Lq[0])]['jump_ratio'], 1e-9))
_gr['P2_accumulate'] = p2; _gr['P2_spearman'] = rho_abs_hat; _gr['P2_n_sites'] = len(Lq)
p3 = bool(Lp and all((prof_abs[str(s)]['jump_ratio'] or 0) < DEC['P3']['J_max'] for s in Lp))
_gr['P3_readout_only'] = p3
V_abs = ('P4_once_formed' if p4 else 'P1_single_source' if p1 else 'P2_accumulate' if p2 else 'P3_readout_only' if p3 else 'P0_no_verdict')

# Q 族（绝对 / 相对）
_SD = E['secondary_decisions_amend1']
_Jv = {s: float(prof_abs[str(s)]['jump_ratio']) for s in PROFILE if prof_abs[str(s)]['jump_ratio'] is not None and np.isfinite(prof_abs[str(s)]['jump_ratio'])}
Lqq = [s for s in PROFILE if s in _Jv]
q1 = q2 = q3 = False; q3_l0 = None; rhoQ = None
if len(Lqq) >= 4:
    vals = [_Jv[s] for s in Lqq]
    spread = max(vals) / max(min(vals), 1e-9)
    q1 = bool(spread <= 1.5 and prof_R_ext['cls'] in ('S_STRONG', 'S_WEAK'))
    rhoQ = spearman(vals, list(Lqq))
    q2 = bool(rhoQ <= -0.6 and _Jv[Lqq[0]] >= 1.5 * _Jv[Lqq[-1]])
    for i in range(len(Lqq) - 1):
        s0, s1 = Lqq[i], Lqq[i + 1]
        pre = [_Jv[t] for t in Lqq[:i + 1]]
        if (_Jv[s0] >= 2.0 * _Jv[s1] and _Jv[s0] >= 3.0 and (max(pre) / max(min(pre), 1e-9)) <= 1.5):
            q3 = True; q3_l0 = s0; break
Q_abs = 'Q3_single_layer' if q3 else ('Q1_readout_origin' if q1 else ('Q2_accumulate' if q2 else 'Q0_no_verdict'))
_gr['Q_family'] = dict(Q1=q1, Q2=q2, Q3=q3, Q3_ell0=q3_l0, spearman=rhoQ,
                       spread=(max([_Jv[s] for s in Lqq]) / max(min([_Jv[s] for s in Lqq]), 1e-9)) if Lqq else None,
                       n_sites=len(Lqq))
_gr['Q_abs'] = Q_abs
_Ju = {s: float(prof_rel[str(s)]['jump_ratio']) for s in PROFILE if prof_rel[str(s)]['jump_ratio'] is not None and np.isfinite(prof_rel[str(s)]['jump_ratio'])}
Lu = [s for s in PROFILE if s in _Ju]
q1u = q2u = False; rhoU = None
if len(Lu) >= 4:
    vu = [_Ju[s] for s in Lu]
    q1u = bool(max(vu) / max(min(vu), 1e-9) <= 1.5 and prof_rel[R_SITE]['cls'] in ('S_STRONG', 'S_WEAK'))
    rhoU = spearman(vu, list(Lu))
    q2u = bool(rhoU <= -0.6 and _Ju[Lu[0]] >= 1.5 * _Ju[Lu[-1]])
Q_rel = 'Q1_readout_origin' if q1u else ('Q2_accumulate' if q2u else 'Q0_no_verdict')
_gr['Q_rel'] = Q_rel; _gr['Q_rel_detail'] = dict(Q1=q1u, Q2=q2u, spearman=rhoU)
_gr['Q_agreement'] = ('Q_ROBUST' if Q_rel == Q_abs else 'Q_COORD_DEPENDENT')

# ---- B 族 ----
B1a = bool(ST_abs['hi'] is not None and ST_abs['hi'] < -0.6)
B1b = bool(ST_rel['hi'] is not None and ST_rel['hi'] < -0.6)
B1 = bool(B1a and B1b)
B2 = bool(rho_own_hat is not None and rho_own_hat <= -0.6 and spread_own is not None and spread_own >= 3.0)
F7p = bool(PLO is not None and PHI is not None and max(abs(PLO), abs(PHI)) < 0.6)
B4 = bool(conf_out.get('spearman_boot') and conf_out['spearman_boot'].get('hi') is not None and conf_out['spearman_boot']['hi'] < 0.0)
# B3 分辨力
n_ind = 0; pairs_ind = []
for i in range(len(Lq) - 1):
    a = Jci.get(str(Lq[i])); b = Jci.get(str(Lq[i + 1]))
    if a and b and a['lo'] is not None and b['lo'] is not None:
        if not (a['hi'] < b['lo'] or b['hi'] < a['lo']):
            n_ind += 1; pairs_ind.append((Lq[i], Lq[i + 1]))
B_verdict = ('GATE_POWER_INSUFFICIENT' if not F7p else
             ('Q2_ESTABLISHED_WITH_BAND' if (B1 and B2) else
              ('Q2_COORD_ROBUST_BASIS_AMBIGUOUS' if B1 else 'Q2_BAND_AMBIGUOUS')))
_gr['B_family'] = dict(B1a=B1a, B1b=B1b, B1=B1, B2=B2, B4=B4, F7prime=F7p,
                       n_indistinguishable=n_ind, indistinguishable_pairs=pairs_ind)
_gr['B_abs_boot'] = ST_abs; _gr['B_rel_boot'] = ST_rel; _gr['B_own_boot'] = ST_own
_gr['B_rho_hat'] = dict(abs=rho_abs_hat, rel=rho_rel_hat, own=rho_own_hat)
_gr['B_spread_own'] = spread_own; _gr['B_spread_own_ci'] = spread_ow_ci
_gr['permutation_null'] = dict(lo=PLO, hi=PHI, B=BP)
_gr['J_site_ci'] = Jci

# V 族
ob_agree = 0; _ob = {}
for s in OB_SITES:
    rows = E3[str(s)]['rows']
    d = curve_stats([r['alpha'] for r in rows], [r['dDonor'] / FULL for r in rows], 'ob')
    _ob[str(s)] = d['cls']
    if str(s) in prof_abs and d['cls'] == prof_abs[str(s)]['cls']:
        ob_agree += 1
_gr['V_ownbasis_detail'] = _ob
_thr_ob = max(3, int(np.ceil(0.75 * len(OB_SITES))))   # amend1 A1：4 位点 3/4 -> 全剖面 75%
_gr['V_ownbasis'] = 'BASIS_ROBUST' if ob_agree >= _thr_ob else 'BASIS_SENSITIVE'
_gr['V_ownbasis_agree'] = '%d/%d' % (ob_agree, len(OB_SITES))
_gr['V_ownbasis_thr'] = '%d/%d' % (_thr_ob, len(OB_SITES))
V_readout = prof_R_ext['cls']
_gr['V_readout'] = V_readout
_gr['V_readout_note'] = ('READOUT_ARTIFACT_WARNING' if V_readout in ('S_STRONG', 'S_WEAK') else
                         'READOUT_LINEAR_OR_GRADUAL' if V_readout in ('LINEAR', 'GRADUAL') else 'READOUT_NOT_DECISIVE')

maxabs = max([abs(r['dDonor']) for s in E1 for r in E1[s]] +
             [abs(r['dDonor']) for s in E1b for r in E1b[s]] +
             [abs(r['dDonor']) for s in E3 for r in E3[s]['rows']] + [1e-9])
E4max = max(abs(r['dDonor']) for r in E4)
F1_ok = bool(E4max < 0.10 * maxabs)
offm = sorted(set([round(r['alpha'], 4) for s in E1 for r in E1[s] if (r['pert_rel'] or 0) > PERT_LIM]))

# ---------------- F8 / F9 / F10 ----------------
# F8: E3(L6) 与 E1(L6) 逐位相等
F8_ok = True; F8_max = 0.0
if str(PRIMARY) in E3:
    for a, row in zip(E3[str(PRIMARY)]['rows'], E1[str(PRIMARY)]):
        dv = abs(row['dDonor'] - a['dDonor'])
        F8_max = max(F8_max, dv)
        if dv > 1e-12:
            F8_ok = False
# F9: E1 对 result_phase10 逐位复现
F9_ok = True; F9_max = 0.0; F9_bad = []
E10E1 = R10.get('E1', {})
for s in E1_SITES:
    if str(s) not in E10E1:
        continue
    a_list = E1[str(s)]; b_list = E10E1[str(s)]
    if len(a_list) != len(b_list):
        F9_bad.append((s, 'len')); F9_ok = False; continue
    for xa, xb in zip(a_list, b_list):
        dv = abs(xa['dDonor'] - xb['dDonor'])
        F9_max = max(F9_max, dv)
        if dv > 1e-12:
            F9_ok = False; F9_bad.append((s, xa['alpha'], dv))
# F10: 逐对均值自洽
F10_ok = True; F10_max = 0.0
for s in E1_SITES:
    for row, per in zip(E1[str(s)], E1P[str(s)]):
        if per:
            dv = abs(float(np.mean(per)) - row['dDonor'])
            F10_max = max(F10_max, dv)
            if dv > 1e-9:
                F10_ok = False

el = time.time() - t0

# ---------------- 16. 报告 ----------------
w('')
w('=== 参数化曲线分类（y = dDonor / full_L6，E1 固定基）===')
w('%-5s %-14s %7s %8s %9s %8s %8s %7s %8s' % ('site', 'cls', 'J', 'x*', 'k_log', 'R2_log', 'R2_lin', 'gamma', 'y_sat'))
for s in E1_SITES:
    d = prof_abs[str(s)]
    w('%-5s %-14s %7s %8s %9s %8s %8s %7s %8.3f' % (
        ('R' if s == R_SITE else 'L%d' % s), d['cls'],
        ('%.2f' % d['jump_ratio']) if d['jump_ratio'] is not None else '  n/a',
        ('%.3f' % d['x_star']) if d['x_star'] is not None else '  n/a',
        ('%.2f' % d['k_log']) if d['k_log'] is not None else '  n/a',
        ('%.4f' % d['R2_log']) if d['R2_log'] is not None else '  n/a',
        ('%.4f' % d['R2_lin']) if d['R2_lin'] is not None else '  n/a',
        ('%.2f' % d['gamma']) if d['gamma'] is not None else '  n/a', d['y_sat']))
w('')
w('=== E3 自基全剖面（J_own 与固定基 J 对照）===')
w('%-6s %10s %10s %10s %8s' % ('site', 'J_own', 'J_fixed', 'overlap', 'cls_own'))
for s in OB_SITES:
    jo = J_own[str(s)]
    jf = prof_abs[str(s)]['jump_ratio'] if str(s) in prof_abs else None
    w('%-6s %10s %10s %10.4f %8s' % ('L%d' % s,
                                     ('%.2f' % jo) if jo is not None and np.isfinite(jo) else 'n/a',
                                     ('%.2f' % jf) if jf is not None else 'n/a',
                                     E3[str(s)]['overlap'], _ob.get(str(s), 'n/a')))
w('')
w('=== 噪声带（配对 bootstrap，B=%d；置换零假设 B_perm=%d）===' % (BS, BP))
w('  绝对剂量剖面 rho_hat = %s ; 95%% 区间 = [%s, %s] ; 有效重采样 %d/%d' % (
    ('%.4f' % rho_abs_hat) if rho_abs_hat is not None else 'n/a',
    ('%.4f' % ST_abs['lo']) if ST_abs['lo'] is not None else 'n/a',
    ('%.4f' % ST_abs['hi']) if ST_abs['hi'] is not None else 'n/a', ST_abs['n_ok'], BS))
w('  相对剂量剖面 rho_hat = %s ; 95%% 区间 = [%s, %s] ; 有效重采样 %d/%d' % (
    ('%.4f' % rho_rel_hat) if rho_rel_hat is not None else 'n/a',
    ('%.4f' % ST_rel['lo']) if ST_rel['lo'] is not None else 'n/a',
    ('%.4f' % ST_rel['hi']) if ST_rel['hi'] is not None else 'n/a', ST_rel['n_ok'], BS))
w('  自基全剖面 rho_hat = %s ; 95%% 区间 = [%s, %s] ; spread=%.3f (带 %s)' % (
    ('%.4f' % rho_own_hat) if rho_own_hat is not None else 'n/a',
    ('%.4f' % ST_own['lo']) if ST_own['lo'] is not None else 'n/a',
    ('%.4f' % ST_own['hi']) if ST_own['hi'] is not None else 'n/a',
    spread_own if spread_own is not None else float('nan'),
    ('[%.2f, %.2f]' % (spread_ow_ci['lo'], spread_ow_ci['hi'])) if spread_ow_ci else 'n/a'))
w('  置换零假设 95%% 带 = [%s, %s]  (F7\' 要求 |界| < 0.6)' % (
    ('%.4f' % PLO) if PLO is not None else 'n/a', ('%.4f' % PHI) if PHI is not None else 'n/a'))
w('  相邻位点 J 区间重叠对数 = %d %s' % (n_ind, pairs_ind if pairs_ind else ''))
w('')
w('=== 判决 ===')
w('  Lq(J有效, E1) = %s (n=%d)' % (Lq, len(Lq)))
w('  V_abs = %s [封存 P 族] ; Q_abs = %s ; Q_rel = %s => %s' % (V_abs, Q_abs, Q_rel, _gr['Q_agreement']))
w('  V_ownbasis = %s (%s) ; V_readout = %s => %s' % (_gr['V_ownbasis'], _gr['V_ownbasis_agree'], V_readout, _gr['V_readout_note']))
w('  B1a(绝对 < -0.6)=%s  B1b(相对 < -0.6)=%s  B1=%s' % (B1a, B1b, B1))
w('  B2(自基 rho<=-0.6 且 spread>=3)=%s  (rho=%s spread=%s)' % (B2,
    ('%.4f' % rho_own_hat) if rho_own_hat is not None else 'n/a', ('%.3f' % spread_own) if spread_own else 'n/a'))
w('  B4(确认集 hi<0)=%s  F7\'=%s' % (B4, F7p))
w('  ==> B 族裁决 = %s' % B_verdict)
w('')
w('=== 装置自检 ===')
w('  F1(rho)=%.4f ; E4_max=%.4f / maxabs=%.3f = %.4f ; F1_ok=%s' % (rho_abs_hat if rho_abs_hat else float('nan'), E4max, maxabs, E4max / max(maxabs, 1e-9), F1_ok))
w('  F6 逐位复现=%s ; E0b drift=%s ; n6 drift=%s' % (E0['dDonor'] == FULL_REF, AD, N6_DRIFT))
w('  F8 E3(L6)==E1(L6) : ok=%s max|d|=%.3e' % (F8_ok, F8_max))
w('  F9 E1 对 Phase10 逐位复现 : ok=%s max|d|=%.3e bad=%s' % (F9_ok, F9_max, F9_bad[:5]))
w('  F10 逐对均值自洽 : ok=%s max|d|=%.3e' % (F10_ok, F10_max))
w('  离流形(pert_rel>%.2f): %s' % (PERT_LIM, offm if offm else 'NONE'))
w('  total %.1fs' % el)
sys.stdout.flush()

# ---------------- 17. 落盘 ----------------
res = dict(
    phase=11, name='N2h1-alpha-4/own_basis_full_profile+noise_band', model=MODEL, smoke=SMOKE,
    elapsed_s=round(el, 1),
    layers=dict(L=L, primary=PRIMARY, pre=PRE, n_heads=NH, head_dim=HD, norm_type=type(NORM).__name__),
    sites=dict(depth=DEPTH, profile=PROFILE, readout=R_SITE, e1=E1_SITES, own_basis=OB_SITES,
               floor=FL_SITES, conf=CF_SITES),
    panel=dict(discovery=len(DISC), confirmation=len(CONF), usable_pairs=len(PAIRS)),
    dose_coord=dict(mean_n6=N6_MEAN, mean_n6_ref_phase9=N6_REF, n6_drift=bool(N6_DRIFT),
                    rbar_ell=RBAR_ELL, r_R=R_R),
    subspace=dict(sing_U6=[float(x) for x in SV6], n_classes=nA6, shape=list(U6.shape)),
    base=dict(ok=(not bad_base), recip_mean=float(np.mean([b['sr0'] for b in BASE.values()])),
              donor_mean=float(np.mean([b['sd0'] for b in BASE.values()]))),
    full_L6=FULL, full_ref_phase9=FULL_REF, bit_replication=bool(E0['dDonor'] == FULL_REF),
    E0=E0, E0b=E0b,
    E1=E1, E1b=E1b, E3=E3, E4=E4, E5=conf_out, E6=E6,
    E1_pairs=E1P, E1b_pairs=E1bP, E3_pairs=E3P,
    profile_abs=prof_abs, profile_rel=prof_rel, profile_R_ext=prof_R_ext,
    E3_verdict=dict(J_own={k: (None if not np.isfinite(v) else float(v)) for k, v in J_own.items()},
                    rho_hat=rho_own_hat, spread=spread_own, spread_ci=spread_ow_ci,
                    boot=ST_own, L=Lo, overlap={str(s): E3[str(s)]['overlap'] for s in OB_SITES}),
    bootstrap_band=dict(B=BS, abs=ST_abs, rel=ST_rel, own=ST_own, rho_hat=dict(abs=rho_abs_hat, rel=rho_rel_hat, own=rho_own_hat),
                        J_site_ci=Jci, n_indistinguishable=n_ind, indistinguishable_pairs=pairs_ind),
    permutation_null=dict(B=BP, lo=PLO, hi=PHI),
    x_star_rel=XSTAR_REL, r_R=R_R,
    decisions=_gr,
    verdict=dict(V_abs=V_abs, Q_abs=Q_abs, Q_rel=Q_rel, Q_agreement=_gr['Q_agreement'],
                 V_ownbasis=_gr['V_ownbasis'], V_ownbasis_agree=_gr['V_ownbasis_agree'],
                 V_readout=V_readout, V_readout_note=_gr['V_readout_note'],
                 B1a=B1a, B1b=B1b, B1=B1, B2=B2, B4=B4, F7prime=F7p, B_verdict=B_verdict),
    floors=dict(F1_ok=F1_ok, E4_max=E4max, maxabs_all=maxabs,
                F3_dev=f3, F3_ok=bool(all(v < 1e-2 for v in f3.values())),
                F6_ok=bool(E0['dDonor'] == FULL_REF),
                F8_ok=bool(F8_ok), F8_max=F8_max,
                F9_ok=bool(F9_ok), F9_max=F9_max,
                F10_ok=bool(F10_ok), F10_max=F10_max),
    off_manifold_alphas=offm,
    drift_flags=dict(n6_drift=bool(N6_DRIFT), anchor_drift=bool(AD), off_manifold=offm),
    seal_sha8=sha(SEAL)[:8], exec_sha8=sha(EXEC)[:8],
)
io.open(RESULT, 'w', encoding='utf-8').write(json.dumps(res, ensure_ascii=False, indent=1))
io.open(REPORT, 'w', encoding='utf-8').write('\n'.join(lines))
print('DONE ->', REPORT, '|', RESULT)
