# -*- coding: utf-8 -*-
"""
Phase 10 / N2h1-alpha-3 : 软阈值的深度定位
=========================================================================
预注册：tests/deepseek_temp/Phase10/N2h1a3_design_seal.json（观测前冻结）
执行冻结：tests/deepseek_temp/Phase10/execution_phase10.json

要回答的问题（Phase 9 §8 死线）：
  Phase 9 已定：L6 内部无阈值增益（amp 平坦），行为曲线是 x*~0.6 的 S 形，
  但只把非线性定位到「L6 之后」这一区间。本 Phase 用同一 D1 剂量探针沿深度逐层复用，
  输出 J(l) / x*(l) / y_sat(l) 剖面，判定：
    P4 L6->L7 一次成形 / P1 单层产生后传递 / P2 逐层累积 / P3 残差链无阈值
  并以位点 R（最终 LayerNorm 之后）作否证探针。

用法：python run_phase10.py smoke | python run_phase10.py formal
"""
import os, sys, io, json, time, hashlib
import numpy as np
import torch

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P10T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase10')
EXEC = os.path.join(P10T, 'execution_phase10.json')
SEAL = os.path.join(P10T, 'N2h1a3_design_seal.json')
REPORT = os.path.join(P10T, 'n2h1a3_report_qwen3-4b.txt')
RESULT = os.path.join(P10T, 'result_phase10.json')
SMOKE = os.environ.get('SMOKE', '0') == '1'
if SMOKE:
    _S = os.path.join(P10T, 'smoke')
    os.makedirs(_S, exist_ok=True)
    REPORT = os.path.join(_S, 'n2h1a3_report_qwen3-4b.txt')
    RESULT = os.path.join(_S, 'result_phase10.json')

lines = []


def w(s=''):
    lines.append(str(s)); print(s); sys.stdout.flush()


E = json.load(io.open(EXEC, encoding='utf-8'))
S = json.load(io.open(SEAL, encoding='utf-8'))

MODEL = E['model']
MDIR = os.path.join(ROOT, 'models', 'hf', MODEL)
TMPL = E['template']
SUP_ID = {k: int(v) for k, v in E['sup_id'].items()}
SUPS = E['classes']
PRIMARY = E['primary_layer']
PRE = E['pre_layer']
NH, HD = E['n_heads'], E['head_dim']
RNG = np.random.default_rng(E['seed'])
RBAR9 = float(E['rbar_ref_from_phase9'])
N6_REF = float(E['mean_n6_ref_from_phase9'])
N6_TOL = float(E['n6_drift_tol'])
FULL_REF = float(E['full_ref_from_phase9'])
D1A_REF = float(E['anchor_ref_d1a_from_phase9'])
ANCHOR_TOL = float(E['anchor_drift_tol'])
PERT_LIM = float(E['off_manifold_pert_rel'])
CL = E['classifier']
DEC = E['decision']
DEPTH = list(E['depth_sites'])
PROFILE = list(E['profile_sites'])      # = [6] + depth_sites（amend1 A2：含 L6 参照点）
R_SITE = E['readout_site']
OB_SITES = list(E['own_basis_sites'])
FL_SITES = list(E['floor_sites'])
CF_SITES = list(E['conf_sites'])
F3_SITES = list(E['f3_sites'])
E1_SITES = PROFILE + [R_SITE]


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


w('=== Phase 10 / N2h1-alpha-3 : 软阈值的深度定位 ===')
w('smoke=%s ; time %s' % (SMOKE, time.strftime('%Y-%m-%d %H:%M:%S')))
w('seal sha8 %s ; exec sha8 %s' % (sha(SEAL)[:8], sha(EXEC)[:8]))
w('exec inherits panel from %s (sha256 %s)' % (E['inherits_panel_from'], E['inherits_panel_sha256'][:16]))
w('exec inherits numbers from %s (sha256 %s)' % (E['inherits_numbers_from'], E['inherits_numbers_sha256'][:16]))
w('config_sha256_match %s (expect %s)' % (
    sha(os.path.join(MDIR, 'config.json')) == E['config_sha256'], E['config_sha256'][:12]))

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
    # 冒烟也要让 U6 满秩：取 6 个类各自的第一个发现实例（否则 rank<5，投影退化为低维）
    DISC = [DISC[i] for i in (0, 4, 8, 12, 16, 20)]
    CONF = []
    DEPTH = DEPTH[:3]
    PROFILE = [PRIMARY] + DEPTH
    E1_SITES = PROFILE + [R_SITE]
    OB_SITES = OB_SITES[:2]
    CF_SITES = []
    _need = set()
    for p in PAIRS_ALL:
        if p[0] in [x[0] for x in DISC]:
            _need.add(p[0]); _need.add(p[2])
    INST_ALL = [(a, b) for (a, b) in INST_ALL if a in _need]
    w('SMOKE panel: discovery=%d ; captured=%s ; depth=%s ; ob=%s' %
      (len(DISC), [x[0] for x in INST_ALL], DEPTH, OB_SITES))


def _keep_sites(lst):
    """冒烟下把位点清单裁到已采集的位点，避免 KeyError；正式运行为恒等操作。"""
    return [s for s in lst if s == R_SITE or s in PROFILE]


F3_SITES = _keep_sites(F3_SITES)
FL_SITES = _keep_sites(FL_SITES)
OB_SITES = _keep_sites(OB_SITES)
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
H_SITES = sorted(set(PROFILE + [PRIMARY]))   # 含 L6 输出（E0/E0b 锚点位点）
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
w('mean||P_U6(diff6)|| = %.4f (Phase 9 参照 %.4f, drift=%s)' % (N6_MEAN, N6_REF, N6_DRIFT))
w('相对剂量 r_ell = mean||u6||/||h_ell_recip|| : %s' %
  '  '.join('L%d:%.4f' % (s, RBAR_ELL[s]) for s in H_SITES))
w('  [口径声明] r_ell 是本 Phase 新定义的『注入向量占该位点残差范数的比例』。'
  'Phase 9 的 rbar=%.5f 是 mean||P_U6(diff5)||/mean||P_U6(diff6)||（相邻两层之间），两者不是同一个量，不作数值比对。' % RBAR9)
w('  r_R = %.5f  （R 位点只有深度位点的约 1/3 => 需要 E6 扩展网格，见 amend1 A5）' % R_R)
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


# ---------------- 7. 剂量执行器 ----------------
def _acc(lg, B, rw, rs, dw, ds):
    sid_r, sid_d = ids_of(rw)[0], ids_of(dw)[0]
    return (score_of(lg, ds, sid_d) - B['sd0'],
            score_of(lg, rs, sid_r) - B['sr0'],
            1 if rank_of(lg, ds, sid_d) == 1 else 0)


def dose_abs(site, alpha_list, pairs, label=''):
    out = []
    for a in alpha_list:
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


def dose_rel(site, arel_list, pairs):
    out = []
    for a in arel_list:
        dd = 0.0; r1 = 0; n = 0
        for (rw, rs, dw, ds, sw) in pairs:
            if rw not in VEC:
                continue
            V, B = VEC[rw], BASE[rw]
            h0, nh0 = base_state(V, site)
            lg = fwd_patch(TMPL % rw, site, torch.tensor(h0 + a * nh0 * V['unit6'], device='cuda'))
            x1, x2, x3 = _acc(lg, B, rw, rs, dw, ds)
            dd += x1; r1 += x3; n += 1
        out.append(dict(alpha_rel=float(a), dDonor=dd / max(n, 1), rank1=r1 / max(n, 1), n=n))
    return out


DISC_P = [p for p in PAIRS if p[0] in [x[0] for x in DISC]]
CONF_P = [p for p in PAIRS if p[0] in [x[0] for x in CONF]]

# ---------------- 8. 锚点 ----------------
w('')
w('--- E0 锚点：S_L6out, alpha=1（内建跨 Phase 复现点）---')
_gl = grid_of('E0_anchor_L6')
E0 = dose_abs(PRIMARY, _gl, DISC_P)[0]
w('  dDonor=%+.15f ; 参照 %.15f ; 逐位相等=%s' %
  (E0['dDonor'], FULL_REF, (E0['dDonor'] == FULL_REF)))
if not SMOKE:
    assert E0['dDonor'] == FULL_REF, 'F6 失败：E0 未逐位复现 Phase 9 的 full'
FULL = float(E0['dDonor'])
w('full_L6 := %.15f' % FULL)

w('')
w('--- E0b 读出口径锚点：S_L6out, alpha=rbar9=%.5f（应对齐 Phase 9 D1a=%+.4f）---' % (RBAR9, D1A_REF))
E0b = dose_abs(PRIMARY, [RBAR9], DISC_P)[0]
AD = abs(E0b['dDonor'] - D1A_REF) > ANCHOR_TOL
w('  dDonor=%+8.4f ; ref %+.4f ; |d|=%.4f ; drift=%s' % (E0b['dDonor'], D1A_REF, abs(E0b['dDonor'] - D1A_REF), AD))
sys.stdout.flush()

# ---------------- 9. E1 绝对剂量深度剖面 ----------------
w('')
w('--- E1 绝对剂量深度剖面（h_ell + alpha * P_U6(diff6)，discovery n=%d）---' % len(DISC_P))
G1 = grid_of('E1_depth_abs')
E1 = {}
for s in E1_SITES:
    r = dose_abs(s, G1, DISC_P)
    E1[str(s)] = r
    w('  L%-3s %s' % (s, '  '.join('a=%.3f dD=%+8.3f pr=%.2f' % (x['alpha'], x['dDonor'], x['pert_rel']) for x in r)))
sys.stdout.flush()

# ---------------- 10. E1b 相对剂量深度剖面 ----------------
w('')
w('--- E1b 相对剂量深度剖面（h_ell + a_rel*||h_ell||*unit(u6)，混淆控制）---')
G2 = grid_of('E1b_depth_rel', 2)
E1b = {}
for s in E1_SITES:
    r = dose_rel(s, G2, DISC_P)
    E1b[str(s)] = r
    w('  L%-3s %s' % (s, '  '.join('r=%.2f dD=%+8.3f' % (x['alpha_rel'], x['dDonor']) for x in r)))
sys.stdout.flush()

# ---------------- 10b. E6 读数位点扩展网格（amend1 A5）----------------
w('')
w('--- E6 读数位点 R 扩展网格（r_R 只有深度位点的约 1/3，同网格打不动）---')
G6 = grid_of('E6_readout_grid')
E6 = dose_abs(R_SITE, G6, DISC_P)
for r in E6:
    w('  R  a=%7.3f  dD=%+8.3f  rank1=%.3f  pert_rel=%.3f' % (r['alpha'], r['dDonor'], r['rank1'], r['pert_rel']))
sys.stdout.flush()

# ---------------- 11. E3 自基稳健性 ----------------w('')
w('--- E3 自基稳健性（h_ell + alpha * P_U{ell}(diff_ell)）---')
G3 = grid_of('E3_own_basis')
E3 = {}
for s in OB_SITES:
    Uell, SVell, nA = est_U(s + 1, DISC)
    Mc = Uell @ AO.T
    svc = np.linalg.svd(Mc, compute_uv=False)
    ov = float(np.sum(svc ** 2) / AO.shape[0])
    out = []
    for a in G3:
        dd = 0.0; n = 0
        for (rw, rs, dw, ds, sw) in DISC_P:
            if rw not in VEC:
                continue
            V, B = VEC[rw], BASE[rw]
            u_own = proj(V['d_ell'][s], Uell)
            lg = fwd_patch(TMPL % rw, s, torch.tensor(V['h_ell'][s] + a * u_own, device='cuda'))
            dd += score_of(lg, ds, ids_of(dw)[0]) - B['sd0']; n += 1
        out.append(dict(alpha=float(a), dDonor=dd / max(n, 1), n=n))
    E3[str(s)] = dict(rows=out, principal_cos=[float(x) for x in svc], overlap=ov,
                      n_classes=nA, sing=[float(x) for x in SVell])
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

# ---------------- 13. 参数化曲线分类器 ----------------
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
    # 线性拟合（x >= 0.10）
    mf = xs >= 0.10
    xf, yf = xs[mf], ys[mf]
    R2l = None
    if len(xf) >= 3:
        b1, b0 = np.polyfit(xf, yf, 1)
        R2l = float(1.0 - np.sum((yf - (b0 + b1 * xf)) ** 2) / max(np.sum((yf - yf.mean()) ** 2), 1e-12))
    # 幂律
    gam = None
    if len(xf) >= 3 and np.all(yf > 0):
        g, _lc = np.polyfit(np.log(xf), np.log(yf), 1)
        gam = float(g)
    # logistic 三参数（A 固定为 max(y)）
    R2g, k_log, x_star = None, None, None
    A = float(np.max(ys))
    if A > 1e-9:
        kk = np.arange(CL['logistic_k_min'], CL['logistic_k_max'] + 1e-9, CL['logistic_k_step'])
        x0 = np.arange(xs.min(), xs.max() + 1e-9, CL['logistic_x0_step'])
        P = A / (1.0 + np.exp(-(kk[:, None, None] * (xs[None, None, :] - x0[None, :, None]))))  # (nk, nx0, npts)
        SSE = ((P - ys[None, None, :]) ** 2).sum(axis=2)
        ij = np.unravel_index(int(np.argmin(SSE)), SSE.shape)
        sse = float(SSE[ij])
        SSt = float(np.sum((ys - ys.mean()) ** 2))
        R2g = float(1.0 - sse / max(SSt, 1e-12))
        k_log = float(kk[ij[0]]); x_star = float(x0[ij[1]])
    det.update(R2_log=R2g, k_log=k_log, x_star=x_star, R2_lin=R2l, gamma=gam,
               y_sat=float(np.max(np.abs(ys))), y_sat_signed=float(np.max(ys)))
    # 分类（阈值见 seal.curve_classifier_parameterized）
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
# R 位点的扩展网格剖面（amend1 A5）
prof_R_ext = curve_stats([r['alpha'] for r in E6], [r['dDonor'] / FULL for r in E6], 'Rext')
# 跨位点可比的相对剂量半饱和点 x*_rel(l) = x*_abs(l) * r_l
XSTAR_REL = {}
for s in E1_SITES:
    xs = prof_abs[str(s)]['x_star']
    rl = R_R if s == R_SITE else RBAR_ELL.get(s)
    XSTAR_REL[str(s)] = (xs * rl) if (xs is not None and rl) else None

# ---------------- 14. 判据 ----------------
Lp = [s for s in PROFILE if prof_abs[str(s)]['cls'] != 'UNREACH']
_gr = {}

# P4
win = [s for s in Lp if DEC['P4']['window'][0] <= s <= DEC['P4']['window'][1]]
Ja = prof_abs['7']['jump_ratio'] if '7' in prof_abs else None
p4 = bool(win and len(win) >= 2 and Ja is not None and Ja >= DEC['P4']['J_anchor'] and
          all(prof_abs[str(s)]['jump_ratio'] is not None and prof_abs[str(s)]['jump_ratio'] >= DEC['P4']['J_anchor'] for s in win))
_gr['P4_once_formed'] = p4
_gr['P4_window'] = win
_gr['P4_J_window'] = [prof_abs[str(s)]['jump_ratio'] for s in win]

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
_gr['P1_single_source'] = p1
_gr['P1_ell0'] = ell0

# P2（amend1 A4：丢弃 J 无效的位点，不用占位值）
p2 = False; rho = None
_Jp = [(s, prof_abs[str(s)]['jump_ratio']) for s in Lp]
_Jp = [(s, float(j)) for (s, j) in _Jp if j is not None and np.isfinite(j)]
if len(_Jp) >= 4:
    rho = spearman([j for _, j in _Jp], [s for s, _ in _Jp])
    p2 = bool(rho >= DEC['P2']['spearman_min'] and
              _Jp[-1][1] >= DEC['P2']['ratio_min'] * max(_Jp[0][1], 1e-9))
_gr['P2_accumulate'] = p2
_gr['P2_spearman'] = rho
_gr['P2_n_sites'] = len(_Jp)
_gr['P2_J_first_last'] = [(_Jp[0][0], _Jp[0][1]), (_Jp[-1][0], _Jp[-1][1])] if _Jp else None

# P3
p3 = bool(Lp and all((prof_abs[str(s)]['jump_ratio'] or 0) < DEC['P3']['J_max'] for s in Lp))
_gr['P3_readout_only'] = p3

if p4:
    V_abs = 'P4_once_formed'
elif p1:
    V_abs = 'P1_single_source'
elif p2:
    V_abs = 'P2_accumulate'
elif p3:
    V_abs = 'P3_readout_only'
else:
    V_abs = 'P0_no_verdict'


# ---- amend1 A3：方向修正的次级判据族 Q（与封存 P 族并列报告，不替换）----
_SD = E['secondary_decisions_amend1']
Jmap = {s: prof_abs[str(s)]['jump_ratio'] for s in PROFILE}
_Jv = {s: float(j) for s, j in Jmap.items() if j is not None and np.isfinite(j)}
Lq = [s for s in PROFILE if s in _Jv]
q1 = q2 = q3 = False; q3_l0 = None; rhoQ = None
if len(Lq) >= 4:
    vals = [_Jv[s] for s in Lq]
    spread = max(vals) / max(min(vals), 1e-9)
    q1 = bool(spread <= 1.5 and prof_R_ext['cls'] in ('S_STRONG', 'S_WEAK'))
    rhoQ = spearman(vals, list(Lq))
    q2 = bool(rhoQ <= -0.6 and _Jv[Lq[0]] >= 1.5 * _Jv[Lq[-1]])
    for i in range(len(Lq) - 1):
        s0, s1 = Lq[i], Lq[i + 1]
        pre = [_Jv[t] for t in Lq[:i + 1]]
        if (_Jv[s0] >= 2.0 * _Jv[s1] and _Jv[s0] >= 3.0 and
                (max(pre) / max(min(pre), 1e-9)) <= 1.5):
            q3 = True; q3_l0 = s0; break
Q_abs = 'Q3_single_layer' if q3 else ('Q1_readout_origin' if q1 else ('Q2_accumulate' if q2 else 'Q0_no_verdict'))
_gr['Q_family'] = dict(Q1=q1, Q2=q2, Q3=q3, Q3_ell0=q3_l0, spearman=rhoQ,
                       spread=max([_Jv[s] for s in Lq]) / max(min([_Jv[s] for s in Lq]), 1e-9) if Lq else None,
                       n_sites=len(Lq))
_gr['Q_abs'] = Q_abs
_gr['Q_rules_text'] = _SD

# V_rel：同一规则作用于 (x_rel, y)
Lr = [s for s in PROFILE if prof_rel[str(s)]['cls'] != 'UNREACH']
p4r = bool([s for s in Lr if 7 <= s <= 10] and len([s for s in Lr if 7 <= s <= 10]) >= 2 and
           prof_rel['7']['jump_ratio'] is not None and prof_rel['7']['jump_ratio'] >= DEC['P4']['J_anchor'] and
           all(prof_rel[str(s)]['jump_ratio'] is not None and prof_rel[str(s)]['jump_ratio'] >= DEC['P4']['J_anchor']
               for s in Lr if 7 <= s <= 10))
p3r = bool(Lr and all((prof_rel[str(s)]['jump_ratio'] or 0) < DEC['P3']['J_max'] for s in Lr))
p2r = False; rhor = None
if len(Lr) >= 4:
    Jr = [prof_rel[str(s)]['jump_ratio'] for s in Lr]
    Jr = [1e6 if j is None else j for j in Jr]
    rhor = spearman(Jr, list(Lr))
    p2r = bool(rhor >= DEC['P2']['spearman_min'] and Jr[-1] >= DEC['P2']['ratio_min'] * max(Jr[0], 1e-9))
if p4r:
    V_rel = 'P4_once_formed'
elif p2r:
    V_rel = 'P2_accumulate'
elif p3r:
    V_rel = 'P3_readout_only'
else:
    V_rel = 'P0_no_verdict'
_gr['V_rel_raw'] = V_rel
_gr['V_rel'] = 'ROBUST' if V_rel == V_abs else 'COORD_DEPENDENT'

# Q 族在相对剂量坐标上的镜像（同为 amend1 次级判据）
_Ju = {s: prof_rel[str(s)]['jump_ratio'] for s in PROFILE}
_Ju = {s: float(j) for s, j in _Ju.items() if j is not None and np.isfinite(j)}
Lu = [s for s in PROFILE if s in _Ju]
q1u = q2u = False; rhoU = None
if len(Lu) >= 4:
    vu = [_Ju[s] for s in Lu]
    q1u = bool(max(vu) / max(min(vu), 1e-9) <= 1.5 and prof_rel[R_SITE]['cls'] in ('S_STRONG', 'S_WEAK'))
    rhoU = spearman(vu, list(Lu))
    q2u = bool(rhoU <= -0.6 and _Ju[Lu[0]] >= 1.5 * _Ju[Lu[-1]])
Q_rel = 'Q1_readout_origin' if q1u else ('Q2_accumulate' if q2u else 'Q0_no_verdict')
_gr['Q_rel'] = Q_rel
_gr['Q_rel_detail'] = dict(Q1=q1u, Q2=q2u, spearman=rhoU)
_gr['Q_agreement'] = ('Q_ROBUST' if Q_rel == Q_abs else 'Q_COORD_DEPENDENT')

# V_ownbasis
ob_agree = 0
_ob = {}
for s in OB_SITES:
    rows = E3[str(s)]['rows']
    d = curve_stats([r['alpha'] for r in rows], [r['dDonor'] / FULL for r in rows], 'ob')
    _ob[str(s)] = d['cls']
    if d['cls'] == prof_abs[str(s)]['cls']:
        ob_agree += 1
_gr['V_ownbasis_detail'] = _ob
_gr['V_ownbasis'] = 'BASIS_ROBUST' if ob_agree >= max(3, len(OB_SITES) - 1) else 'BASIS_SENSITIVE'
_gr['V_ownbasis_agree'] = '%d/%d' % (ob_agree, len(OB_SITES))

# V_readout（amend1 A5：由 E6 扩展网格给出；同网格结果单列记账）
V_readout = prof_R_ext['cls']
_gr['V_readout'] = V_readout
_gr['V_readout_same_grid_cls'] = prof_abs[R_SITE]['cls']
_gr['V_readout_note'] = ('READOUT_ARTIFACT_WARNING' if V_readout in ('S_STRONG', 'S_WEAK') else
                         'READOUT_LINEAR_OR_GRADUAL' if V_readout in ('LINEAR', 'GRADUAL') else
                         'READOUT_NOT_DECISIVE')

# 组合裁决文本
combo_txt = {
    'P4_once_formed': '软阈值在 L6->L7 之间一次性成形：写入窗与阈值窗紧邻，L7 的注意力/MLP 是成形者',
    'P1_single_source': '阈值由可指名层 ell0=%s 产生，之后被递送' % str(ell0),
    'P2_accumulate': '阈值是整条深层栈的累积效应：层=软门 须改为 栈=软门',
    'P3_readout_only': '残差链上不存在软阈值：线性链 + 非线性读数',
    'P0_no_verdict': '剖面无单一判决：J(l) 在 2-3 之间，须带噪声带重测',
}.get(V_abs, 'n/a')
if _gr['V_readout_note'] == 'READOUT_ARTIFACT_WARNING':
    combo_txt += ' ｜ 但位点 R 亦呈 %s => 软阈值可能是读数/解嵌几何假象（否证条款触发）' % V_readout
if _gr['V_rel'] == 'COORD_DEPENDENT':
    combo_txt += ' ｜ 相对剂量坐标给出 %s => 结论受范数混淆影响，降级为描述性' % V_rel
Q_txt = {
    'Q1_readout_origin': 'Q 族（方向修正）：J(l) 全程近常数且 R 位点亦呈 S 形 => 非线性集中在最后一个被探测层之下（读数端）',
    'Q2_accumulate': 'Q 族（方向修正）：J 越浅越大（随经过层数累积）=> S 形由多层逐步塑形',
    'Q3_single_layer': 'Q 族（方向修正）：J 在 l0=%s 与 l0+1 之间断崖 => 非线性由 layer %s 产生（注入位点 l0 与 l0+1 之差 = 经过 layer l0+1）' % (str(q3_l0), str(q3_l0 + 1) if isinstance(q3_l0, int) else 'n/a'),
    'Q0_no_verdict': 'Q 族（方向修正）：无单一判决',
}.get(Q_abs, 'n/a')

maxabs = max([abs(r['dDonor']) for s in E1 for r in E1[s]] +
             [abs(r['dDonor']) for s in E1b for r in E1b[s]] +
             [abs(r['dDonor']) for s in E3 for r in E3[s]['rows']] + [1e-9])
E4max = max(abs(r['dDonor']) for r in E4)
F1_ok = bool(E4max < 0.10 * maxabs)
offm = sorted(set([round(r['alpha'], 4) for s in E1 for r in E1[s] if (r['pert_rel'] or 0) > PERT_LIM]))

# ---------------- 15. E5 确认集 ----------------
conf_out = {}
if CF_SITES and CONF_P:
    w('')
    w('--- E5 确认集验带（n=%d，位点 %s）---' % (len(CONF_P), CF_SITES))
    conf_out['full_conf'] = None
    conf_out['sites'] = {}
    for s in CF_SITES:
        rows = dose_abs(s, G1, CONF_P)
        d = curve_stats([r['alpha'] for r in rows], [r['dDonor'] / FULL for r in rows], 'conf')
        conf_out['sites'][str(s)] = dict(rows=rows, cls=d['cls'], J=d['jump_ratio'], x_star=d['x_star'],
                                         y_sat=d['y_sat'], R2_log=d['R2_log'], same_cls=bool(d['cls'] == prof_abs[str(s)]['cls']))
        w('  L%-3s %s  J=%s x*=%s y_sat=%.3f 与发现集同判=%s' %
          (s, d['cls'], ('%.2f' % d['jump_ratio']) if d['jump_ratio'] is not None else 'n/a',
           ('%.3f' % d['x_star']) if d['x_star'] is not None else 'n/a', d['y_sat'],
           conf_out['sites'][str(s)]['same_cls']))
    conf_out['same_cls_all'] = bool(all(v['same_cls'] for v in conf_out['sites'].values()))
    conf_out['same_cls_frac'] = float(np.mean([1.0 if v['same_cls'] else 0.0 for v in conf_out['sites'].values()]))

el = time.time() - t0

# ---------------- 16. 报告剖面表 ----------------
w('')
w('=== 参数化曲线分类（y = dDonor / full_L6）===')
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
        ('%.2f' % d['gamma']) if d['gamma'] is not None else '  n/a',
        d['y_sat']))
_dR = prof_R_ext
w('%-5s %-14s %7s %8s %9s %8s %8s %7s %8.3f' % (
    'R*', _dR['cls'],
    ('%.2f' % _dR['jump_ratio']) if _dR['jump_ratio'] is not None else '  n/a',
    ('%.3f' % _dR['x_star']) if _dR['x_star'] is not None else '  n/a',
    ('%.2f' % _dR['k_log']) if _dR['k_log'] is not None else '  n/a',
    ('%.4f' % _dR['R2_log']) if _dR['R2_log'] is not None else '  n/a',
    ('%.4f' % _dR['R2_lin']) if _dR['R2_lin'] is not None else '  n/a',
    ('%.2f' % _dR['gamma']) if _dR['gamma'] is not None else '  n/a',
    _dR['y_sat']))
w('  R* = 位点 R 的扩展网格（amend1 A5，alpha 到 16）；其余行 = E1 同网格')
w('')
w('=== 相对剂量剖面（E1b）===')
w('%-5s %-14s %7s %8s %8s' % ('site', 'cls', 'J', 'x*', 'y_sat'))
for s in E1_SITES:
    d = prof_rel[str(s)]
    w('%-5s %-14s %7s %8s %8.3f' % (
        ('R' if s == R_SITE else 'L%d' % s), d['cls'],
        ('%.2f' % d['jump_ratio']) if d['jump_ratio'] is not None else '  n/a',
        ('%.3f' % d['x_star']) if d['x_star'] is not None else '  n/a', d['y_sat']))
w('')
w('=== 判决 ===')
w('  L (非 UNREACH 的剖面位点，含 L6) = %s' % Lp)
w('  V_abs = %s  [封存 P 族]' % V_abs)
w('  Q_abs = %s  [amend1 方向修正族] ; Q_rel = %s => %s' % (Q_abs, Q_rel, _gr['Q_agreement']))
w('  V_rel = %s (原始 %s) %s' % (V_rel, _gr['V_rel_raw'], _gr['V_rel']))
w('  V_ownbasis = %s (%s)' % (_gr['V_ownbasis'], _gr['V_ownbasis_agree']))
w('  V_readout = %s [扩展网格 R*] => %s ; 同网格 R 为 %s' %
  (V_readout, _gr['V_readout_note'], _gr['V_readout_same_grid_cls']))
w('  P4=%s P1=%s(%s) P2=%s(rho=%s, n=%d) P3=%s' % (p4, p1, ell0, p2, rho, _gr['P2_n_sites'], p3))
w('  Q1=%s Q2=%s Q3=%s(%s) Q_spread=%s' % (q1, q2, q3, q3_l0, _gr['Q_family']['spread']))
w('  组合裁决[封存] : %s' % combo_txt)
w('  组合裁决[修正] : %s' % Q_txt)
w('')
w('=== 剖面数组（便于逐点复核）===')
w('  J_abs  : %s' % '  '.join('%s:%s' % ('R' if s == R_SITE else ('L%d' % s),
                                          ('%.2f' % prof_abs[str(s)]['jump_ratio']) if prof_abs[str(s)]['jump_ratio'] is not None else 'n/a')
                              for s in E1_SITES))
w('  x*_abs : %s' % '  '.join('%s:%s' % ('R' if s == R_SITE else ('L%d' % s),
                                          ('%.3f' % prof_abs[str(s)]['x_star']) if prof_abs[str(s)]['x_star'] is not None else 'n/a')
                              for s in E1_SITES))
w('  ysat   : %s' % '  '.join('%s:%.3f' % ('R' if s == R_SITE else ('L%d' % s), prof_abs[str(s)]['y_sat'])
                              for s in E1_SITES))
w('  J_rel  : %s' % '  '.join('%s:%s' % ('R' if s == R_SITE else ('L%d' % s),
                                          ('%.2f' % prof_rel[str(s)]['jump_ratio']) if prof_rel[str(s)]['jump_ratio'] is not None else 'n/a')
                              for s in E1_SITES))
w('  ysat_r : %s' % '  '.join('%s:%.3f' % ('R' if s == R_SITE else ('L%d' % s), prof_rel[str(s)]['y_sat'])
                              for s in E1_SITES))
w('  x*_rel : %s   (= x*_abs * r_ell，跨位点可比)' % '  '.join(
    '%s:%s' % ('R' if s == R_SITE else ('L%d' % s),
               ('%.3f' % XSTAR_REL[str(s)]) if XSTAR_REL[str(s)] is not None else 'n/a') for s in E1_SITES))
w('  r_ell  : %s' % '  '.join('%s:%.4f' % ('R' if s == R_SITE else ('L%d' % s),
                                          R_R if s == R_SITE else RBAR_ELL[s]) for s in E1_SITES))
w('')
w('地板: E4_max=%.4f ; maxabs_all=%.3f ; 比=%.4f ; F1_ok=%s' % (E4max, maxabs, E4max / max(maxabs, 1e-9), F1_ok))
w('离流形警告（pert_rel > %.2f 的 alpha）: %s' % (PERT_LIM, offm if offm else 'NONE'))
w('F3 = %s ; F6 逐位复现 = %s ; E0b drift = %s ; n6 drift = %s' % (f3, E0['dDonor'] == FULL_REF, AD, N6_DRIFT))
w('total %.1fs' % el)

# ---------------- 17. 落盘 ----------------
res = dict(
    phase=10, name='N2h1-alpha-3/soft_threshold_depth_locating', model=MODEL, smoke=SMOKE,
    elapsed_s=round(el, 1),
    layers=dict(L=L, primary=PRIMARY, pre=PRE, n_heads=NH, head_dim=HD, norm_type=type(NORM).__name__),
    sites=dict(depth=DEPTH, profile=PROFILE, readout=R_SITE, e1=E1_SITES, own_basis=OB_SITES,
               floor=FL_SITES, conf=CF_SITES),
    panel=dict(discovery=len(DISC), confirmation=len(CONF)),
    dose_coord=dict(mean_n6=N6_MEAN, mean_n6_ref_phase9=N6_REF, n6_drift=bool(N6_DRIFT),
                    rbar_ell=RBAR_ELL, r_R=float(np.mean([VEC[rw]['n6'] / max(VEC[rw]['nhR'], 1e-9) for rw in VEC]))),
    subspace=dict(sing_U6=[float(x) for x in SV6], n_classes=nA6, shape=list(U6.shape)),
    base=dict(ok=(not bad_base), recip_mean=float(np.mean([b['sr0'] for b in BASE.values()])),
              donor_mean=float(np.mean([b['sd0'] for b in BASE.values()])),
              donor_rank1_frac=float(np.mean([1.0 if BASE[rw]['rd0'] == 1 else 0.0 for rw in BASE]))),
    full_L6=FULL, full_ref_phase9=FULL_REF, bit_replication=bool(E0['dDonor'] == FULL_REF),
    E0=E0, E0b=E0b,
    E1={k: v for k, v in E1.items()}, E1b=E1b, E3=E3, E4=E4, E5=conf_out, E6=E6,
    profile_abs=prof_abs, profile_rel=prof_rel, profile_R_ext=prof_R_ext,
    x_star_rel=XSTAR_REL, r_R=R_R,
    decisions=_gr,
    verdict=dict(V_abs=V_abs, V_rel=V_rel, V_rel_raw=_gr['V_rel_raw'], V_ownbasis=_gr['V_ownbasis'],
                 V_readout=V_readout, V_readout_note=_gr['V_readout_note'], combo=combo_txt,
                 Q_abs=Q_abs, Q_rel=Q_rel, Q_agreement=_gr['Q_agreement'], Q_combo=Q_txt,
                 P4=p4, P1=p1, P1_ell0=ell0, P2=p2, P2_spearman=rho, P3=p3),
    amend1='N2h1a3-amend1 (A1 E1b 网格 4->7 点 / A2 剖面加 L6 参照 / A3 新增方向修正判据族 Q / A4 代码修正；'
           '面板与全部封存阈值未动)',
    floors=dict(F1_ok=F1_ok, E4_max=E4max, maxabs_all=maxabs, F2_ok=(not bad_base),
                F3_dev=f3, F3_ok=bool(all(v < 1e-2 for v in f3.values())), F6_ok=bool(E0['dDonor'] == FULL_REF),
                F6b_anchor_drift=bool(AD)),
    off_manifold_alphas=offm,
    drift_flags=dict(n6_drift=bool(N6_DRIFT), anchor_drift=bool(AD), off_manifold=offm),
    seal_sha8=sha(SEAL)[:8], exec_sha8=sha(EXEC)[:8],
)
io.open(RESULT, 'w', encoding='utf-8').write(json.dumps(res, ensure_ascii=False, indent=1))
io.open(REPORT, 'w', encoding='utf-8').write('\n'.join(lines))
print('DONE ->', REPORT, '|', RESULT)
