# -*- coding: utf-8 -*-
"""
Phase 12 / N2h1-alpha-5 : 逐层残差替换 + 层贡献分配
=========================================================================
预注册：tests/deepseek_temp/Phase12/N2h1a5_design_seal.json（观测前冻结）
执行冻结：tests/deepseek_temp/Phase12/execution_phase12.json

要回答的问题（Phase 11 §8 死线，最高优先）：
  Phase 8-11 的全部结论都建立在同一探针族上：注入 u6 = P_U6(diff6)（一个在 L6 算出、
  rank 5、跨层固定的向量）。Phase 11 的 B3 又证明 J 的精度不足以排序位点。
  本 Phase 换探针族：把受体在 ell 处的末位残差【替换】为供体的末位残差
  （h_ell_recip + alpha * diff_ell，alpha 属于 [0,1] 是替换比例，alpha=1 即完全替换），
  在 18 个剖面位点 + R 上给出剂量-响应曲线族，得到
    (i)  结晶曲线 recover(ell) = dDonor(alpha=1)/FULL_SWAP
    (ii) 替换坐标下的锐度剖面 J_swap(ell)
  用 G2 集中度判据（死线原文『3 层内贡献 >= 60% 的 spread』）判定
  『栈=软门』是【少层主导】还是【逐层累积】；用 G3 秩相关判定替换坐标与注入坐标是否同形。

用法：python run_phase12.py smoke | python run_phase12.py formal
"""
import os, sys, io, json, time, hashlib
import numpy as np
import torch

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P12T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase12')
EXEC = os.path.join(P12T, 'execution_phase12.json')
SEAL = os.path.join(P12T, 'N2h1a5_design_seal.json')
REPORT = os.path.join(P12T, 'n2h1a5_report_qwen3-4b.txt')
RESULT = os.path.join(P12T, 'result_phase12.json')
SMOKE = os.environ.get('SMOKE', '0') == '1'
if SMOKE:
    _S = os.path.join(P12T, 'smoke')
    os.makedirs(_S, exist_ok=True)
    REPORT = os.path.join(_S, 'n2h1a5_report_qwen3-4b.txt')
    RESULT = os.path.join(_S, 'result_phase12.json')

lines = []


def w(s=''):
    lines.append(str(s)); print(s); sys.stdout.flush()


E = json.load(io.open(EXEC, encoding='utf-8'))
S = json.load(io.open(SEAL, encoding='utf-8'))

# 防御：execution 必需字段自检（Phase 11 首版 gen 曾漏字段导致 KeyError）
_REQ = ['inherits_panel_sha256', 'inherits_panel10_sha256', 'inherits_panel8_sha256',
        'phase11_result_for_F13', 'phase11_result_sha256', 'bootstrap', 'swap',
        'profile_sites', 'swap_sites', 'swap_rel_sites', 'overshoot_sites', 'arms',
        'classifier', 'g_family', 'decision', 'result_keys']
_miss = [k for k in _REQ if k not in E]
assert not _miss, 'execution_phase12.json 缺字段: %s（请检查 gen 脚本）' % _miss

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
G = E['g_family']
BOOT = E['bootstrap']
DEPTH = list(E['depth_sites'])
PROFILE = list(E['profile_sites'])              # 18 个位点
SWAP_SITES = list(E['swap_sites'])
RL_SITES = list(E['swap_rel_sites'])
OV_SITES = list(E['overshoot_sites'])
R_SITE = E['readout_site']
FL_SITES = list(E['floor_sites'])
CF_SITES = list(E['conf_sites'])
F3_SITES = list(E['f3_sites'])


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


w('=== Phase 12 / N2h1-alpha-5 : 逐层残差替换 + 层贡献分配 ===')
w('smoke=%s ; time %s' % (SMOKE, time.strftime('%Y-%m-%d %H:%M:%S')))
w('seal sha8 %s ; exec sha8 %s' % (sha(SEAL)[:8], sha(EXEC)[:8]))
w('exec inherits panel(phase11) sha256 %s' % E['inherits_panel_sha256'][:16])
w('exec inherits panel(phase10) sha256 %s' % E['inherits_panel10_sha256'][:16])
w('exec inherits panel(phase8)  sha256 %s' % E['inherits_panel8_sha256'][:16])
w('config_sha256_match %s (expect %s)' % (
    sha(os.path.join(MDIR, 'config.json')) == E['config_sha256'], E['config_sha256'][:12]))

# ---- F13 参照：Phase 11 结果（G3 的 J_inject 来源） ----
R11P = os.path.join(ROOT, E['phase11_result_for_F13'])
assert sha(R11P) == E['phase11_result_sha256'], 'F13 前置：result_phase11.json 已漂移'
R11 = json.load(io.open(R11P, encoding='utf-8'))
J_INJECT = {}
for _k, _v in R11['profile_abs'].items():
    try:
        J_INJECT[int(_k)] = _v['jump_ratio']
    except (TypeError, ValueError):
        pass   # payload 里含 'R' 等非数字键（SMOKE 抓到）

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
assert max(DEPTH) < L - 1, 'depth_sites 超出层数（须剔末层）'

PAIRS_ALL = [tuple(p) for p in E['pairs_all']]
DISC = [tuple(x) for x in E['discovery']]
CONF = [tuple(x) for x in E['confirmation']]
INST_ALL = [tuple(x) for x in E['instances_all']]

if SMOKE:
    DISC = [DISC[i] for i in (0, 4, 8, 12, 16, 20)]
    CONF = []
    PROFILE = [6, 7, 8]
    SWAP_SITES = list(PROFILE)
    RL_SITES = [7]
    OV_SITES = [6]
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
w('profile_sites (%d) = %s' % (len(PROFILE), PROFILE))
w('panel: discovery=%d confirmation=%d (all=%d)' % (len(DISC), len(CONF), len(INST_ALL)))
sys.stdout.flush()

# ---------------- 1. 采集（全 hidden states + norm 输出 + logits）----------------
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
w('capture done %d instances in %.1fs (hidden_levels=%d)' %
  (len(CAP), time.time() - t_cap, CAP[INST_ALL[0][0]][0].shape[0]))
if SMOKE:
    wd0 = INST_ALL[0][0]
    HH, hR, lg = CAP[wd0]
    w('SMOKE assert: HH%s hR%s logits%s nan=%s' % (HH.shape, hR.shape, lg.shape, bool(np.isnan(HH).any())))
    assert HH.shape[1] == HID and hR.shape[0] == HID and not np.isnan(HH).any()
PAIRS = [p for p in PAIRS_ALL if p[0] in CAP and p[2] in CAP]
w('usable pairs (recipient & donor captured) = %d / %d' % (len(PAIRS), len(PAIRS_ALL)))

H6 = PRIMARY + 1   # hidden_states[7] = L6 输出
sys.stdout.flush()


# ---------------- 2. 类子空间 U6（discovery only；只用于 E0 锚与诊断）----------------
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

# ---------------- 4. 每对的替换向量与各位点基准态 ----------------
H_SITES = sorted(set(PROFILE + [PRIMARY]))   # 含 L6 输出
VEC = {}
for (rw, rs, dw, ds, sw) in PAIRS:
    h6r, h6d = CAP[rw][0][H6].astype(np.float32), CAP[dw][0][H6].astype(np.float32)
    d6 = h6d - h6r
    u6 = proj(d6, AO)
    n6 = float(np.linalg.norm(u6))
    d = dict(u6=u6, n6=n6, unit6=(u6 / max(n6, 1e-9)),
             hR=CAP[rw][1].astype(np.float32), nhR=float(np.linalg.norm(CAP[rw][1])),
             h_ell={}, hd_ell={}, nh_ell={}, d_ell={})
    for s in H_SITES:
        hr = CAP[rw][0][s + 1].astype(np.float32)
        hd = CAP[dw][0][s + 1].astype(np.float32)
        d['h_ell'][s] = hr
        d['hd_ell'][s] = hd
        d['nh_ell'][s] = float(np.linalg.norm(hr))
        d['d_ell'][s] = hd - hr
    # R 位点（最终 LayerNorm 输出）
    d['hR_donor'] = CAP[dw][1].astype(np.float32)
    d['d_R'] = d['hR_donor'] - d['hR']
    d['n_dR'] = float(np.linalg.norm(d['d_R']))
    VEC[rw] = d

N6_MEAN = float(np.mean([VEC[rw]['n6'] for rw in VEC]))
N6_DRIFT = abs(N6_MEAN - N6_REF) > N6_TOL

# q_ell = ||diff_ell|| / ||h_ell_recip||（满替换的相对幅度）；q_R 同法
Q_ELL = {}
for s in PROFILE:
    Q_ELL[str(s)] = float(np.mean([np.linalg.norm(VEC[rw]['d_ell'][s]) / max(VEC[rw]['nh_ell'][s], 1e-9)
                                   for rw in VEC]))
Q_R = float(np.mean([VEC[rw]['n_dR'] / max(VEC[rw]['nhR'], 1e-9) for rw in VEC]))
R_R = float(np.mean([VEC[rw]['n6'] / max(VEC[rw]['nhR'], 1e-9) for rw in VEC]))

# proj_share_u6(ell) = ||P_U6(diff_ell)|| / ||diff_ell||（diff_ell 有多少落在类别轴上）
PSU = {}
for s in PROFILE:
    vals = []
    for rw in VEC:
        dv = VEC[rw]['d_ell'][s]
        nd = float(np.linalg.norm(dv))
        if nd > 1e-9:
            vals.append(float(np.linalg.norm(proj(dv, AO))) / nd)
    PSU[str(s)] = float(np.mean(vals))
PSU_R = float(np.mean([float(np.linalg.norm(proj(VEC[rw]['d_R'], AO))) / max(VEC[rw]['n_dR'], 1e-9)
                       for rw in VEC]))

w('')
w('--- 替换幅度与投影份额 ---')
w('mean||P_U6(diff6)|| = %.4f (Phase 9/10/11 参照 %.4f, drift=%s)' % (N6_MEAN, N6_REF, N6_DRIFT))
w('q_ell (= a_rel_full, 满替换的相对幅度) : %s' %
  '  '.join('L%d:%.3f' % (s, Q_ELL[str(s)]) for s in PROFILE))
w('q_R = %.4f ; r_R = %.4f（注意：r_R 是 u6 口径，与 Phase 9 的 rbar=%.5f 不是同一个量）' % (Q_R, R_R, RBAR9))
w('proj_share_u6(ell) (= ||P_U6(diff_ell)||/||diff_ell||) : %s' %
  '  '.join('L%d:%.3f' % (s, PSU[str(s)]) for s in PROFILE))
w('proj_share_u6(R) = %.4f' % PSU_R)
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


# ---------------- 6. F3 / F12 构造自检 ----------------
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

w('')
w('--- F12 自检：alpha=1 的替换向量必须等于供体贴残差（float32 重建误差范围内）---')
f12 = {}
for site in [6, 7, 20, 34, R_SITE]:
    vals = []
    for rw in VEC:
        V = VEC[rw]
        if site == R_SITE:
            lhs = V['hR'] + 1.0 * V['d_R']
            rhs = V['hR_donor']
        else:
            if site not in V['hd_ell']:
                continue
            lhs = V['h_ell'][site] + 1.0 * V['d_ell'][site]
            rhs = V['hd_ell'][site]
        den = max(float(np.linalg.norm(rhs)), 1e-9)
        vals.append(float(np.linalg.norm(lhs - rhs)) / den)
    f12[str(site)] = float(max(vals)) if vals else 0.0
w('  max relative reconstruction error : %s' %
  '  '.join('%s=%.3e' % (k, v) for k, v in f12.items()))
w('  (alpha=1 的语义由 diff := h(donor) - h(recip) 的构造保证；此处核对 float32 重建误差)')
assert all(v < 1e-5 for v in f12.values()), 'F12 失败：满替换不是精确的供体贴残差'
sys.stdout.flush()


# ---------------- 7. 剂量执行器 ----------------
def _acc(lg, B, rw, rs, dw, ds):
    sid_r, sid_d = ids_of(rw)[0], ids_of(dw)[0]
    return (score_of(lg, ds, sid_d) - B['sd0'],
            score_of(lg, rs, sid_r) - B['sr0'],
            1 if rank_of(lg, ds, sid_d) == 1 else 0)


def _vec_of(V, site):
    return V['d_R'] if site == R_SITE else V['d_ell'][site]


def dose_swap(site, alphas, pairs, per_pair=True):
    """h_ell_recip + alpha * diff_ell（alpha=1 => 完全替换为供体贴）"""
    out = []
    for a in alphas:
        dd = dr = 0.0; r1 = 0; n = 0; per = []; order = []; perts = []
        for (rw, rs, dw, ds, sw) in pairs:
            if rw not in VEC:
                continue
            V, B = VEC[rw], BASE[rw]
            h0, nh0 = base_state(V, site)
            dv = _vec_of(V, site)
            lg = fwd_patch(TMPL % rw, site, torch.tensor(h0 + a * dv, device='cuda'))
            x1, x2, x3 = _acc(lg, B, rw, rs, dw, ds)
            dd += x1; dr += x2; r1 += x3; n += 1
            per.append(float(x1)); order.append(rw)
            perts.append(a * float(np.linalg.norm(dv)) / max(nh0, 1e-9))
        row = dict(alpha=float(a), dDonor=dd / max(n, 1), dRecip=dr / max(n, 1),
                   rank1=r1 / max(n, 1), n=n, pert_rel=float(np.mean(perts)))
        if per_pair:
            row['per_pair'] = per; row['order'] = order
        out.append(row)
    return out


def dose_swap_rel(site, arels, pairs, per_pair=True):
    """h_ell_recip + a_rel * ||h_ell_recip|| * unit(diff_ell)"""
    out = []
    for a in arels:
        dd = 0.0; r1 = 0; n = 0; per = []; order = []
        for (rw, rs, dw, ds, sw) in pairs:
            if rw not in VEC:
                continue
            V, B = VEC[rw], BASE[rw]
            h0, nh0 = base_state(V, site)
            dv = _vec_of(V, site)
            u = dv / max(float(np.linalg.norm(dv)), 1e-9)
            lg = fwd_patch(TMPL % rw, site, torch.tensor(h0 + a * nh0 * u, device='cuda'))
            x1, x2, x3 = _acc(lg, B, rw, rs, dw, ds)
            dd += x1; r1 += x3; n += 1
            per.append(float(x1)); order.append(rw)
        row = dict(alpha_rel=float(a), dDonor=dd / max(n, 1), rank1=r1 / max(n, 1), n=n)
        if per_pair:
            row['per_pair'] = per; row['order'] = order
        out.append(row)
    return out


def dose_inject(site, alphas, pairs):
    """Phase 8-11 的注入口径：h_ell_recip + alpha * u6（仅用于 E0 跨 Phase 锚点）"""
    out = []
    for a in alphas:
        dd = dr = 0.0; r1 = 0; n = 0; perts = []
        for (rw, rs, dw, ds, sw) in pairs:
            if rw not in VEC:
                continue
            V, B = VEC[rw], BASE[rw]
            h0, nh0 = base_state(V, site)
            lg = fwd_patch(TMPL % rw, site, torch.tensor(h0 + a * V['u6'], device='cuda'))
            x1, x2, x3 = _acc(lg, B, rw, rs, dw, ds)
            dd += x1; dr += x2; r1 += x3; n += 1
            perts.append(a * V['n6'] / max(nh0, 1e-9))
        out.append(dict(alpha=float(a), dDonor=dd / max(n, 1), dRecip=dr / max(n, 1),
                        rank1=r1 / max(n, 1), n=n, pert_rel=float(np.mean(perts))))
    return out


DISC_P = [p for p in PAIRS if p[0] in [x[0] for x in DISC]]
CONF_P = [p for p in PAIRS if p[0] in [x[0] for x in CONF]]

# ---------------- 8. E0 跨 Phase 锚点 ----------------
w('')
w('--- E0 锚点：S_L6out, alpha=1（注入口径，内建跨 Phase 复现点）---')
_gl = grid_of('E0_anchor_L6')
E0 = dose_inject(PRIMARY, _gl, DISC_P)[0]
w('  dDonor=%+.15f ; 参照 %.15f ; 逐位相等=%s' %
  (E0['dDonor'], FULL_REF, (E0['dDonor'] == FULL_REF)))
if not SMOKE:
    assert E0['dDonor'] == FULL_REF, 'F6 失败：E0 未逐位复现 Phase 9 的 full'
w('full_L6 := %.15f' % float(E0['dDonor']))

# ---------------- 9. FULL_SWAP（供体自身前向，零额外前向）----------------
FS_PAIR = {}
for (rw, rs, dw, ds, sw) in PAIRS:
    FS_PAIR[rw] = float(score_of(CAP[dw][2], ds, ids_of(dw)[0]) - BASE[rw]['sd0'])
FULL_SWAP = float(np.mean([FS_PAIR[rw] for (rw, rs, dw, ds, sw) in DISC_P]))
FS_ORDER = [rw for (rw, rs, dw, ds, sw) in DISC_P]
FS_VEC = np.array([FS_PAIR[rw] for rw in FS_ORDER], float)
w('')
w('--- FULL_SWAP（供体自身前向 dDonor，零额外前向）---')
w('  FULL_SWAP = %+.6f  (n=%d ; min=%+.3f max=%+.3f)' %
  (FULL_SWAP, len(FS_VEC), float(FS_VEC.min()), float(FS_VEC.max())))
sys.stdout.flush()

# ---------------- 10. E2 主臂：逐层替换曲线（逐对落盘）----------------
w('')
w('--- E2 逐层替换剖面（h_ell + alpha*diff_ell，%d 位点，discovery n=%d）---' % (len(SWAP_SITES), len(DISC_P)))
G2A = grid_of('E2_swap_curve')
E2 = {}
E2P = {}
for s in SWAP_SITES:
    r = dose_swap(s, G2A, DISC_P, per_pair=True)
    E2[str(s)] = r
    E2P[str(s)] = [x.get('per_pair', []) for x in r]
    w('  L%-3s %s' % (s, '  '.join('a=%.2f dD=%+7.3f' % (x['alpha'], x['dDonor']) for x in r)))
sys.stdout.flush()

# ---------------- 11. E6 读数位点替换 ----------------
w('')
w('--- E6 读数位点 R 的替换曲线 ---')
G6 = grid_of('E6_readout_swap')
E6 = dose_swap(R_SITE, G6, DISC_P, per_pair=True)
E6P = [x.get('per_pair', []) for x in E6]
for r in E6:
    w('  R  a=%.2f dD=%+8.3f rank1=%.3f pert_rel=%.3f' % (r['alpha'], r['dDonor'], r['rank1'], r['pert_rel']))
# F11：alpha=1 处必须逐位等于 FULL_SWAP（构造决定点）
_r1 = [r for r in E6 if abs(r['alpha'] - 1.0) < 1e-12]
F11_ok = True; F11_max = float('nan'); F11_val = float('nan')
if _r1:
    F11_val = float(_r1[0]['dDonor'])
    F11_max = abs(F11_val - FULL_SWAP)
    F11_ok = bool(F11_max <= 1e-6)
w('  F11 构造恒等：E6[R](alpha=1) = %+.15f vs FULL_SWAP = %+.15f ; |d|=%.3e ; ok=%s' %
  (F11_val, FULL_SWAP, F11_max, F11_ok))
if not SMOKE:
    assert F11_ok, 'F11 失败：R 位点满替换未复现 FULL_SWAP（实现缺陷，HALT）'
sys.stdout.flush()

# ---------------- 12. E2b 相对坐标 ----------------
w('')
w('--- E2b 相对坐标（h_ell + a_rel*||h_ell||*unit(diff_ell)，位点 %s）---' % RL_SITES)
G2B = grid_of('E2b_swap_rel', 2)
E2b = {}
E2bP = {}
for s in RL_SITES:
    r = dose_swap_rel(s, G2B, DISC_P, per_pair=True)
    E2b[str(s)] = r
    E2bP[str(s)] = [x.get('per_pair', []) for x in r]
    w('  L%-3s %s' % (s, '  '.join('r=%.2f dD=%+7.3f' % (x['alpha_rel'], x['dDonor']) for x in r)))
sys.stdout.flush()

# ---------------- 13. E3 超量（alpha>1，描述性）----------------
w('')
w('--- E3 超量外推（alpha>1，描述性饱和检查）---')
G3O = grid_of('E3_overshoot', 2)
E3 = {}
for s in OV_SITES:
    r = dose_swap(s, G3O, DISC_P, per_pair=False)
    E3[str(s)] = r
    w('  L%-3s %s' % (s, '  '.join('a=%.2f dD=%+7.3f' % (x['alpha'], x['dDonor']) for x in r)))
sys.stdout.flush()

# ---------------- 14. E4 地板（全空间范数匹配随机方向）----------------
w('')
w('--- E4 地板（全空间范数匹配随机方向，2 draws）---')
_drw = E['arms']['E4_floor_swap']['draws']
E4 = []
for s in FL_SITES:
    dd, n = 0.0, 0
    for (rw, rs, dw, ds, sw) in DISC_P:
        if rw not in VEC:
            continue
        V, B = VEC[rw], BASE[rw]
        h0, _nh = base_state(V, s)
        nd = float(np.linalg.norm(_vec_of(V, s)))
        for _ in range(_drw):
            z = RNG.standard_normal(HID).astype(np.float32)
            v = z / max(float(np.linalg.norm(z)), 1e-9) * nd
            lg = fwd_patch(TMPL % rw, s, torch.tensor(h0 + v, device='cuda'))
            dd += score_of(lg, ds, ids_of(dw)[0]) - B['sd0']; n += 1
    E4.append(dict(site=s, alpha=1.0, dDonor=dd / max(n, 1), n=n))
    w('  L%-3s dDonor=%+8.4f (n=%d)' % (s, E4[-1]['dDonor'], n))
sys.stdout.flush()

# ---------------- 15. 分类器（与 Phase 10/11 逐字相同）----------------
def _rank(a):
    a = np.asarray(a, float)
    order = np.argsort(a)
    r = np.empty(len(a), float)
    r[order] = np.arange(len(a), dtype=float)
    return r


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2:
        return None
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
        sse = float(SSE[ij]); SSt = float(np.sum((ys - ys.mean()) ** 2))
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


prof_swap, prof_swap_rel = {}, {}
for s in SWAP_SITES:
    rows = E2[str(s)]
    prof_swap[str(s)] = curve_stats([r['alpha'] for r in rows], [r['dDonor'] / FULL_SWAP for r in rows], 'swap')
for s in RL_SITES:
    rows = E2b[str(s)]
    prof_swap_rel[str(s)] = curve_stats([r['alpha_rel'] for r in rows], [r['dDonor'] / FULL_SWAP for r in rows], 'swaprel')
prof_R_swap = curve_stats([r['alpha'] for r in E6], [r['dDonor'] / FULL_SWAP for r in E6], 'swapR')

# ---------------- 16. 结晶曲线 recover 与 G 族 ----------------
def _recover_of(rows, alpha_val=1.0):
    for r in rows:
        if abs(r['alpha'] - alpha_val) < 1e-12:
            return float(r['dDonor'] / FULL_SWAP)
    return float('nan')


RECOVER = {s: _recover_of(E2[str(s)]) for s in SWAP_SITES}
RECOVER_R = _recover_of(E6)
SPAN = RECOVER[SWAP_SITES[-1]] - RECOVER[SWAP_SITES[0]]


def cross_depth(sites, vals, target):
    for i in range(len(sites) - 1):
        v0, v1 = vals[i], vals[i + 1]
        if v0 == v1:
            continue
        if (v0 - target) * (v1 - target) <= 0:
            t = (target - v0) / (v1 - v0)
            return float(sites[i] + t * (sites[i + 1] - sites[i]))
    return None


CURVE = [RECOVER[s] for s in SWAP_SITES]
INC = [CURVE[i + 1] - CURVE[i] for i in range(len(CURVE) - 1)]
TOTAL = CURVE[-1] - CURVE[0]
W = int(G['G2_concentration']['window'])
WINS = [float(sum(INC[i:i + W])) for i in range(len(INC) - W + 1)] if len(INC) >= W else []
G0_ok = bool(SPAN >= G['G0_precondition']['span_min'])
if TOTAL > 1e-9:
    TOP3_SHARE = float(max(WINS) / TOTAL) if WINS else None
    TOP3_MASS = float(sum(sorted(INC, reverse=True)[:3]) / TOTAL)
    MAX_SHARE = float(max(INC) / TOTAL)
    FLATNESS = float(max(INC) / max(np.mean(INC), 1e-12))
else:
    TOP3_SHARE = TOP3_MASS = MAX_SHARE = FLATNESS = None
RNORM = [c / CURVE[-1] for c in CURVE] if CURVE[-1] > 1e-9 else None
X_HALF = cross_depth(SWAP_SITES, RNORM, 0.5) if RNORM else None
D10 = cross_depth(SWAP_SITES, RNORM, 0.1) if RNORM else None
D90 = cross_depth(SWAP_SITES, RNORM, 0.9) if RNORM else None
SPAN_10_90 = (D90 - D10) if (D10 is not None and D90 is not None) else None
RHO_REC = spearman(CURVE, SWAP_SITES) if len(CURVE) >= 4 else None

# G1
if not G0_ok:
    G1 = 'G1_NA'
elif X_HALF is not None and SPAN_10_90 is not None and X_HALF <= 20 and SPAN_10_90 <= 14:
    G1 = 'G1a_crystallized'
elif SPAN_10_90 is not None and SPAN_10_90 >= 24:
    G1 = 'G1b_distributed'
else:
    G1 = 'G1_mid'

# G2
if not G0_ok or TOP3_SHARE is None:
    G2 = 'G2_NA'
elif TOP3_SHARE >= G['G2_concentration']['G2a_top3_share_min']:
    G2 = 'G2a_few_layer_dominant'
elif TOP3_SHARE <= G['G2_concentration']['G2b_top3_share_max'] and MAX_SHARE <= G['G2_concentration']['G2b_max_share_max']:
    G2 = 'G2b_layerwise_accumulate'
else:
    G2 = 'G2_mid'

# G3
_JG = [(s, prof_swap[str(s)]['jump_ratio'], J_INJECT.get(s)) for s in SWAP_SITES]
_JG = [(s, a, b) for (s, a, b) in _JG if a is not None and np.isfinite(a) and b is not None and np.isfinite(b)]
if len(_JG) >= 4:
    RHO_JG = spearman([x[1] for x in _JG], [x[2] for x in _JG])
else:
    RHO_JG = None
if RHO_JG is None:
    G3 = 'G3_NA'
elif RHO_JG >= G['G3_profile_shape']['rho_JG_same_min']:
    G3 = 'G3_same_gradient'
elif RHO_JG <= G['G3_profile_shape']['rho_JG_indep_max']:
    G3 = 'G3_independent'
else:
    G3 = 'G3_weak'

# ---------------- 17. bootstrap 与置换零假设 ----------------
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


BS = BOOT['B'] if not SMOKE else 200
BP = BOOT['B_perm'] if not SMOKE else 200
BRNG = np.random.default_rng(int(BOOT['seed']))     # bootstrap 专用独立流

PM_swap = None; PM_R = None
boot = dict(B=BS, recover_ci={}, J_ci={}, rho_recover=None, top3_share_ci=None,
            max_share_ci=None, R_ci=None, n_ok=0)
if not SMOKE and len(SWAP_SITES) >= 4 and DISC_P:
    PM_swap = np.stack([np.array(E2P[str(s)], dtype=float) for s in SWAP_SITES], 0)   # (nS, nA, nP)
    PM_R = np.array(E6P, dtype=float)                                                # (nA, nP)
    xs_sw = np.array([r['alpha'] for r in E2[str(SWAP_SITES[0])]], float)
    nS, nA, nP = PM_swap.shape
    a1 = int(np.argmin(np.abs(xs_sw - 1.0)))
    rec_b = np.full((BS, nS), np.nan)
    J_b = np.full((BS, nS), np.nan)
    rho_b = np.full(BS, np.nan)
    t3_b = np.full(BS, np.nan)
    mx_b = np.full(BS, np.nan)
    recR_b = np.full(BS, np.nan)
    for b in range(BS):
        idx = BRNG.integers(0, nP, nP)
        fs_b = float(FS_VEC[idx].mean())
        if abs(fs_b) < 1e-9:
            continue
        Y = PM_swap[:, :, idx].mean(axis=2) / fs_b          # (nS, nA)
        rec_b[b] = Y[:, a1]
        for i in range(nS):
            J_b[b, i] = J_only(xs_sw, Y[i])
        rr = rec_b[b]
        if np.all(np.isfinite(rr)) and len(rr) >= 4:
            rho_b[b] = spearman(rr, SWAP_SITES)
            inc = np.diff(rr); tot = rr[-1] - rr[0]
            if tot > 1e-9 and len(inc) >= W:
                wins = [sum(inc[i:i + W]) for i in range(len(inc) - W + 1)]
                t3_b[b] = max(wins) / tot
                mx_b[b] = max(inc) / tot
        YR = PM_R[:, idx].mean(axis=1) / fs_b
        recR_b[b] = YR[a1]

    def _ci(v):
        v = v[np.isfinite(v)]
        if len(v) < 10:
            return dict(lo=None, hi=None, med=None, n_ok=int(len(v)))
        lo, hi = np.percentile(v, [2.5, 97.5])
        return dict(lo=float(lo), hi=float(hi), med=float(np.median(v)), n_ok=int(len(v)))

    boot['rho_recover'] = dict(hat=RHO_REC, **_ci(rho_b))
    boot['top3_share_ci'] = _ci(t3_b)
    boot['max_share_ci'] = _ci(mx_b)
    boot['R_ci'] = dict(hat=RECOVER_R, **_ci(recR_b))
    boot['n_ok'] = int(np.isfinite(rho_b).sum())
    for i, s in enumerate(SWAP_SITES):
        boot['recover_ci'][str(s)] = dict(hat=float(RECOVER[s]), **_ci(rec_b[:, i]))
        boot['J_ci'][str(s)] = dict(hat=float(prof_swap[str(s)]['jump_ratio'])
                                     if prof_swap[str(s)]['jump_ratio'] is not None else None,
                                     **_ci(J_b[:, i]))

perm = np.array([]); PLO = PHI = None
if not SMOKE and len(SWAP_SITES) >= 4:
    perm = np.empty(BP)
    for b in range(BP):
        perm[b] = spearman(BRNG.permutation(CURVE), SWAP_SITES)
    pr = perm[np.isfinite(perm)]
    if len(pr) >= 10:
        PLO, PHI = float(np.percentile(pr, 2.5)), float(np.percentile(pr, 97.5))
F7pp = bool(PLO is not None and PHI is not None and max(abs(PLO), abs(PHI)) < 0.6)

# ---------------- 18. E5 确认集 ----------------
w('')
conf_out = {}
confP = {}
if CF_SITES and CONF_P:
    w('--- E5 确认集（n=%d，位点 %s）---' % (len(CONF_P), CF_SITES))
    G5 = grid_of('E5_conf_swap', 3)
    conf_rec = {}
    for s in CF_SITES:
        rows = dose_swap(s, G5, CONF_P, per_pair=True)
        confP[str(s)] = [x.get('per_pair', []) for x in rows]
        conf_rec[s] = _recover_of(rows)
        d = curve_stats([r['alpha'] for r in rows], [r['dDonor'] / FULL_SWAP for r in rows], 'conf')
        conf_out[str(s)] = dict(rows=rows, cls=d['cls'], J=d['jump_ratio'], recover=float(conf_rec[s]),
                                x_star=d['x_star'], y_sat=d['y_sat'])
        w('  L%-3s recover=%.4f  J=%s cls=%s' % (s, conf_rec[s],
                                                 ('%.2f' % d['jump_ratio']) if d['jump_ratio'] is not None else 'n/a',
                                                 d['cls']))
    RHO_CONF = spearman([conf_rec[s] for s in CF_SITES], CF_SITES) if len(CF_SITES) >= 4 else None
    conf_out['rho_recover'] = RHO_CONF
    # 确认集上的 recover 带
    if not SMOKE and len(CF_SITES) >= 4:
        PMc = np.stack([np.array(confP[str(s)], dtype=float) for s in CF_SITES], 0)
        xsc = np.array([r['alpha'] for r in conf_out[str(CF_SITES[0])]['rows']], float)
        a1c = int(np.argmin(np.abs(xsc - 1.0)))
        nPc = PMc.shape[2]
        rb = np.full(BS, np.nan)
        for b in range(BS):
            idx = BRNG.integers(0, nPc, nPc)
            Yc = PMc[:, :, idx].mean(axis=2) / FULL_SWAP
            rb[b] = spearman(Yc[:, a1c], CF_SITES)
        conf_out['rho_boot'] = _ci(rb)
    G4 = bool(RHO_CONF is not None and RHO_REC is not None and
              (RHO_CONF * RHO_REC > 0) and abs(RHO_CONF) >= G['G4_confirmation']['rho_abs_min'])
    conf_out['G4'] = G4
else:
    w('  (确认集在冒烟下跳过)')
    RHO_CONF = None; G4 = None
sys.stdout.flush()

# ---------------- 19. 组合裁决 ----------------
if not G0_ok:
    G_VERDICT = 'SWAP_UNINFORMATIVE'
elif G2 == 'G2a_few_layer_dominant' and G3 == 'G3_same_gradient':
    G_VERDICT = 'FEW_LAYER_DOMINANT_STACK'
elif G2 == 'G2b_layerwise_accumulate' and G3 == 'G3_same_gradient':
    G_VERDICT = 'LAYERWISE_ACCUMULATE_CONFIRMED'
elif G2 == 'G2a_few_layer_dominant' and G3 != 'G3_same_gradient':
    G_VERDICT = 'FEW_LAYER_DOMINANT_BUT_COORD_SPECIFIC'
elif G2 == 'G2b_layerwise_accumulate' and G3 != 'G3_same_gradient':
    G_VERDICT = 'ACCUMULATE_BUT_COORD_SPECIFIC'
else:
    G_VERDICT = 'ALLOCATION_AMBIGUOUS'

# ---------------- 20. 地板与装置自检 ----------------
maxabs = max([abs(r['dDonor']) for s in E2 for r in E2[s]] +
             [abs(r['dDonor']) for s in E2b for r in E2b[s]] +
             [abs(r['dDonor']) for s in E3 for r in E3[s]] +
             [abs(r['dDonor']) for r in E6] + [1e-9])
E4max = max(abs(r['dDonor']) for r in E4)
F1_ok = bool(E4max < 0.10 * maxabs)
offm = sorted(set([round(r['alpha'], 4) for s in E2 for r in E2[s] if (r['pert_rel'] or 0) > PERT_LIM]))
# F10：逐对均值自洽
F10_ok = True; F10_max = 0.0
for s in SWAP_SITES:
    for row, per in zip(E2[str(s)], E2P[str(s)]):
        if per:
            dv = abs(float(np.mean(per)) - row['dDonor'])
            F10_max = max(F10_max, dv)
            if dv > 1e-9:
                F10_ok = False
# F12 汇总（已在第 6 节算）
F12_ok = bool(all(v < 1e-6 for v in f12.values()))
el = time.time() - t0

# ---------------- 21. 报告 ----------------
w('')
w('=== 结晶曲线 recover(ell) = dDonor(alpha=1) / FULL_SWAP ===')
w('%(s)-6s %(rec)10s %(jsw)10s %(jinj)10s %(cls)14s %(q)8s %(psu)8s' %
  dict(s='site', rec='recover', jsw='J_swap', jinj='J_inject', cls='cls_swap', q='q_ell', psu='psu_u6'))
for s in SWAP_SITES:
    jf = J_INJECT.get(s)
    w('L%-5d %10.4f %10s %10s %-14s %8.3f %8.3f' % (
        s, RECOVER[s],
        ('%.2f' % prof_swap[str(s)]['jump_ratio']) if prof_swap[str(s)]['jump_ratio'] is not None and np.isfinite(prof_swap[str(s)]['jump_ratio']) else 'n/a',
        ('%.2f' % jf) if jf is not None else 'n/a',
        prof_swap[str(s)]['cls'], Q_ELL[str(s)], PSU[str(s)]))
w('R      %10.4f %10s %10s %-14s %8.3f %8.3f' % (
    RECOVER_R,
    ('%.2f' % prof_R_swap['jump_ratio']) if prof_R_swap['jump_ratio'] is not None and np.isfinite(prof_R_swap['jump_ratio']) else 'n/a',
    'n/a', prof_R_swap['cls'], Q_R, PSU_R))
w('')
w('=== G 族判决 ===')
w('  G0 span = %.4f (阈值 %.2f) -> ok=%s' % (SPAN, G['G0_precondition']['span_min'], G0_ok))
w('  rho_recover(depth) = %s  (bootstrap 95%% 带 %s)' % (
    ('%.4f' % RHO_REC) if RHO_REC is not None else 'n/a',
    ('[%.4f, %.4f]' % (boot['rho_recover']['lo'], boot['rho_recover']['hi']))
    if boot['rho_recover'].get('lo') is not None else 'n/a'))
w('  G1 x_half = %s ; span_10_90 = %s -> %s' % (
    ('%.2f' % X_HALF) if X_HALF is not None else 'n/a',
    ('%.2f' % SPAN_10_90) if SPAN_10_90 is not None else 'n/a', G1))
w('  G2 top3_share = %s (带 %s) ; top3_mass = %s ; max_share = %s ; flatness = %s -> %s' % (
    ('%.4f' % TOP3_SHARE) if TOP3_SHARE is not None else 'n/a',
    ('[%.4f, %.4f]' % (boot['top3_share_ci']['lo'], boot['top3_share_ci']['hi']))
    if boot['top3_share_ci'].get('lo') is not None else 'n/a',
    ('%.4f' % TOP3_MASS) if TOP3_MASS is not None else 'n/a',
    ('%.4f' % MAX_SHARE) if MAX_SHARE is not None else 'n/a',
    ('%.3f' % FLATNESS) if FLATNESS is not None else 'n/a', G2))
w('  G3 rho(J_swap, J_inject) = %s (n=%d) -> %s' % (
    ('%.4f' % RHO_JG) if RHO_JG is not None else 'n/a', len(_JG), G3))
w('  G4 确认集 rho = %s -> %s' % (('%.4f' % RHO_CONF) if RHO_CONF is not None else 'n/a', G4))
w('  置换零假设 95%% 带 = [%s, %s] (F7(dbl-prime) 要求 |界| < 0.6) -> %s' % (
    ('%.4f' % PLO) if PLO is not None else 'n/a',
    ('%.4f' % PHI) if PHI is not None else 'n/a', F7pp))
w('  ==> G 族裁决 = %s' % G_VERDICT)
w('')
w('=== 装置自检 ===')
w('  F1 max|E4| = %.4f / maxabs = %.3f = %.4f ; ok=%s' % (E4max, maxabs, E4max / max(maxabs, 1e-9), F1_ok))
w('  F6 E0 逐位复现 = %s ; n6 drift = %s' % (E0['dDonor'] == FULL_REF, N6_DRIFT))
w('  F10 逐对均值自洽 ok=%s max|d|=%.3e' % (F10_ok, F10_max))
w('  F11 R 位点满替换 == FULL_SWAP ok=%s max|d|=%.3e' % (F11_ok, F11_max))
w('  F12 满替换 = 供体贴残差 ok=%s' % F12_ok)
w('  F13 result_phase11 sha 一致 = %s' % (sha(R11P) == E['phase11_result_sha256']))
w('  离流形 alpha (pert_rel>%.2f): %s' % (PERT_LIM, offm if offm else 'NONE'))
w('  total %.1fs' % el)
sys.stdout.flush()

# ---------------- 22. 落盘 ----------------
res = dict(
    phase=12, name=E['name'], model=MODEL, smoke=SMOKE, elapsed_s=round(el, 1),
    layers=dict(L=L, primary=PRIMARY, pre=PRE, n_heads=NH, head_dim=HD, norm_type=type(NORM).__name__),
    sites=dict(depth=DEPTH, profile=PROFILE, readout=R_SITE, swap=SWAP_SITES,
               swap_rel=RL_SITES, overshoot=OV_SITES, floor=FL_SITES, conf=CF_SITES),
    panel=dict(discovery=len(DISC), confirmation=len(CONF), usable_pairs=len(PAIRS)),
    dose_coord=dict(mean_n6=N6_MEAN, mean_n6_ref=N6_REF, n6_drift=bool(N6_DRIFT),
                    q_ell=Q_ELL, q_R=Q_R, r_R=R_R, rbar_ref_phase9=RBAR9),
    diff_norms={str(s): float(np.mean([np.linalg.norm(VEC[rw]['d_ell'][s]) for rw in VEC])) for s in PROFILE},
    proj_share_u6=dict(ell=PSU, R=PSU_R),
    subspace=dict(sing_U6=[float(x) for x in SV6], n_classes=nA6, shape=list(U6.shape)),
    base=dict(ok=(not bad_base), recip_mean=float(np.mean([b['sr0'] for b in BASE.values()])),
              donor_mean=float(np.mean([b['sd0'] for b in BASE.values()]))),
    E0=E0, full_L6=float(E0['dDonor']), full_ref_phase9=FULL_REF,
    bit_replication=bool(E0['dDonor'] == FULL_REF),
    FULL_SWAP=FULL_SWAP, FULL_SWAP_pairs=FS_PAIR,
    E2=E2, E2b=E2b, E3=E3, E4=E4, E6=E6, E5=conf_out,
    E2_pairs=E2P, E2b_pairs=E2bP, E5_pairs={k: v for k, v in (confP if CF_SITES else {}).items()},
    E6_pairs=E6P,
    profile_swap=prof_swap, profile_swap_rel=prof_swap_rel, profile_R_swap=prof_R_swap,
    recover=dict(curve={str(s): float(RECOVER[s]) for s in SWAP_SITES},
                 R=float(RECOVER_R), span=float(SPAN), rho=RHO_REC),
    G_family=dict(
        G0=dict(ok=bool(G0_ok), span=float(SPAN), span_min=float(G['G0_precondition']['span_min'])),
        G1=dict(label=G1, x_half=X_HALF, span_10_90=SPAN_10_90, r_norm=(RNORM or [])),
        G2=dict(label=G2, top3_share=TOP3_SHARE, top3_mass=TOP3_MASS, max_share=MAX_SHARE,
                flatness=FLATNESS, total=TOTAL, inc=INC, window=W),
        G3=dict(label=G3, rho_JG=RHO_JG, n_sites=len(_JG),
                pairs=[(s, a, b) for (s, a, b) in _JG]),
        G4=dict(label=('G4_ok' if G4 else ('G4_fail' if G4 is not None else 'G4_na')),
                rho_conf=RHO_CONF),
        F7pp=bool(F7pp),
    ),
    G_verdict=G_VERDICT,
    bootstrap_band=boot,
    permutation_null=dict(B=BP, lo=PLO, hi=PHI, values=perm.tolist() if perm.size else []),
    floors=dict(F1_ok=bool(F1_ok), E4_max=E4max, maxabs_all=maxabs,
                F3_dev=f3, F3_ok=bool(all(v < 1e-2 for v in f3.values())),
                F6_ok=bool(E0['dDonor'] == FULL_REF),
                F10_ok=bool(F10_ok), F10_max=F10_max,
                F11_ok=bool(F11_ok), F11_max=F11_max, F11_val=F11_val,
                F12_ok=bool(F12_ok), F12_dev=f12,
                F13_ok=bool(sha(R11P) == E['phase11_result_sha256'])),
    off_manifold_alphas=offm,
    drift_flags=dict(n6_drift=bool(N6_DRIFT), off_manifold=offm),
    seal_sha8=sha(SEAL)[:8], exec_sha8=sha(EXEC)[:8],
    inherits_panel_sha8=E['inherits_panel_sha256'][:8],
    phase11_result_sha8=E['phase11_result_sha256'][:8],
)
io.open(RESULT, 'w', encoding='utf-8').write(json.dumps(res, ensure_ascii=False, indent=1))
io.open(REPORT, 'w', encoding='utf-8').write('\n'.join(lines))
print('DONE ->', REPORT, '|', RESULT)
