# -*- coding: utf-8 -*-
"""
Phase 12 / N2h1-alpha-5 : 逐层残差替换 + 层贡献分配   (v2 = 含 amend1)
=========================================================================
预注册：tests/deepseek_temp/Phase12/N2h1a5_design_seal.json（观测前冻结）
修订：  tests/deepseek_temp/Phase12/N2h1a5_design_seal_amend1.json（正式运行前冻结）
执行冻结：tests/deepseek_temp/Phase12/execution_phase12.json

要回答的问题（Phase 11 §8 死线，最高优先）：
  Phase 8-11 的全部结论都建立在同一探针族上：注入 u6 = P_U6(diff6)（在 L6 算出、rank 5、
  跨层固定的向量）。Phase 11 的 B3 又证明 J 的精度不足以排序位点。
  本 Phase 换探针族：把受体在 ell 处的末位残差【替换】为供体的末位残差
  （h_ell_recip + alpha * diff_ell，alpha 属于 [0,1] 是替换比例，alpha=1 即完全替换），
  在 18 个剖面位点 + R 上给出剂量-响应曲线族。

amend1 的指标修正（SMOKE 触发，正式运行前冻结）：
  * recover(alpha=1) 按构造饱和（Phase 11 已证：在 L6 只注入 rank-5 类别轴、alpha=1 即得
    full_L6 = 10.574739583333335；满替换是严格更大的干预）=> 端点量不携带深度信息。
  * 故主量改为曲线的形状量：xhalf(ell)（经验半饱和替换比例）与 J_swap(ell)（锐度）；
    端点量 recover 升格为独立的正向判据 G5（末位状态充分性）。

用法：python run_phase12.py smoke | python run_phase12.py formal
"""
import os, sys, io, json, time, hashlib
import numpy as np
import torch

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P12T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase12')
EXEC = os.path.join(P12T, 'execution_phase12.json')
SEAL = os.path.join(P12T, 'N2h1a5_design_seal.json')
AMEND = os.path.join(P12T, 'N2h1a5_design_seal_amend1.json')
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
A1 = json.load(io.open(AMEND, encoding='utf-8'))

_REQ = ['inherits_panel_sha256', 'inherits_panel10_sha256', 'inherits_panel8_sha256',
        'phase11_result_for_F13', 'phase11_result_sha256', 'bootstrap', 'swap', 'amend1',
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
RNG = np.random.default_rng(E['seed'])
RBAR9 = float(E['rbar_ref_from_phase9'])
N6_REF = float(E['mean_n6_ref_from_phase9'])
N6_TOL = float(E['n6_drift_tol'])
FULL_REF = float(E['full_ref_from_phase9'])
D1A_REF = float(E['anchor_ref_d1a_from_phase9'])
ANCHOR_TOL = float(E['anchor_drift_tol'])
PERT_LIM = float(E['off_manifold_pert_rel'])
CL = E['classifier']
G = E['g_family']
BOOT = E['bootstrap']
DEPTH = list(E['depth_sites'])
PROFILE = list(E['profile_sites'])
SWAP_SITES = list(E['swap_sites'])
RL_SITES = list(E['swap_rel_sites'])
OV_SITES = list(E['overshoot_sites'])
R_SITE = E['readout_site']
FL_SITES = list(E['floor_sites'])
CF_SITES = list(E['conf_sites'])
F3_SITES = list(E['f3_sites'])


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


w('=== Phase 12 / N2h1-alpha-5 : 逐层残差替换 + 层贡献分配 (v2/amend1) ===')
w('smoke=%s ; time %s' % (SMOKE, time.strftime('%Y-%m-%d %H:%M:%S')))
w('seal sha8 %s ; amend1 sha8 %s ; exec sha8 %s' % (sha(SEAL)[:8], sha(AMEND)[:8], sha(EXEC)[:8]))
w('exec amend1 sha256 %s (expect %s)' % (E['amend1']['sha256'][:16], sha(AMEND)[:16]))
w('exec inherits panel(phase11) sha256 %s' % E['inherits_panel_sha256'][:16])
w('exec inherits panel(phase10) sha256 %s' % E['inherits_panel10_sha256'][:16])
w('exec inherits panel(phase8)  sha256 %s' % E['inherits_panel8_sha256'][:16])
w('config_sha256_match %s (expect %s)' % (
    sha(os.path.join(MDIR, 'config.json')) == E['config_sha256'], E['config_sha256'][:12]))

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
H6 = PRIMARY + 1
sys.stdout.flush()


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

H_SITES = sorted(set(PROFILE + [PRIMARY]))
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
    d['hR_donor'] = CAP[dw][1].astype(np.float32)
    d['d_R'] = d['hR_donor'] - d['hR']
    d['n_dR'] = float(np.linalg.norm(d['d_R']))
    VEC[rw] = d

N6_MEAN = float(np.mean([VEC[rw]['n6'] for rw in VEC]))
N6_DRIFT = abs(N6_MEAN - N6_REF) > N6_TOL
Q_ELL = {}
for s in PROFILE:
    Q_ELL[str(s)] = float(np.mean([np.linalg.norm(VEC[rw]['d_ell'][s]) / max(VEC[rw]['nh_ell'][s], 1e-9)
                                   for rw in VEC]))
Q_R = float(np.mean([VEC[rw]['n_dR'] / max(VEC[rw]['nhR'], 1e-9) for rw in VEC]))
R_R = float(np.mean([VEC[rw]['n6'] / max(VEC[rw]['nhR'], 1e-9) for rw in VEC]))
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
w('q_R = %.4f ; r_R = %.4f（r_R 是 u6 口径，与 Phase 9 的 rbar=%.5f 不是同一个量）' % (Q_R, R_R, RBAR9))
w('proj_share_u6(ell) (= ||P_U6(diff_ell)||/||diff_ell||) : %s' %
  '  '.join('L%d:%.3f' % (s, PSU[str(s)]) for s in PROFILE))
w('proj_share_u6(R) = %.4f' % PSU_R)
sys.stdout.flush()


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
assert all(v < 1e-5 for v in f12.values()), 'F12 失败：满替换不是精确的供体贴残差'
sys.stdout.flush()


def _acc(lg, B, rw, rs, dw, ds):
    sid_r, sid_d = ids_of(rw)[0], ids_of(dw)[0]
    return (score_of(lg, ds, sid_d) - B['sd0'],
            score_of(lg, rs, sid_r) - B['sr0'],
            1 if rank_of(lg, ds, sid_d) == 1 else 0)


def _vec_of(V, site):
    return V['d_R'] if site == R_SITE else V['d_ell'][site]


def dose_swap(site, alphas, pairs, per_pair=True):
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

w('')
w('--- E0 锚点：S_L6out, alpha=1（注入口径，内建跨 Phase 复现点）---')
_gl = grid_of('E0_anchor_L6')
E0 = dose_inject(PRIMARY, _gl, DISC_P)[0]
w('  dDonor=%+.15f ; 参照 %.15f ; 逐位相等=%s' %
  (E0['dDonor'], FULL_REF, (E0['dDonor'] == FULL_REF)))
if not SMOKE:
    assert E0['dDonor'] == FULL_REF, 'F6 失败：E0 未逐位复现 Phase 9 的 full'
w('full_L6 := %.15f' % float(E0['dDonor']))

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

w('')
w('--- E6 读数位点 R 的替换曲线 ---')
G6 = grid_of('E6_readout_swap')
E6 = dose_swap(R_SITE, G6, DISC_P, per_pair=True)
E6P = [x.get('per_pair', []) for x in E6]
for r in E6:
    w('  R  a=%.2f dD=%+8.3f rank1=%.3f pert_rel=%.3f' % (r['alpha'], r['dDonor'], r['rank1'], r['pert_rel']))
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

w('')
w('--- E3 超量外推（alpha>1，描述性饱和检查）---')
G3O = grid_of('E3_overshoot', 2)
E3 = {}
for s in OV_SITES:
    r = dose_swap(s, G3O, DISC_P, per_pair=False)
    E3[str(s)] = r
    w('  L%-3s %s' % (s, '  '.join('a=%.2f dD=%+7.3f' % (x['alpha'], x['dDonor']) for x in r)))
sys.stdout.flush()

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


def cross_alpha(xs, ys, frac):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    if len(xs) < 2:
        return None
    ymax = float(np.max(ys))
    if not np.isfinite(ymax) or abs(ymax) < 1e-12:
        return None
    tgt = frac * ymax
    for i in range(len(xs) - 1):
        if ys[i] < tgt <= ys[i + 1]:
            t = (tgt - ys[i]) / (ys[i + 1] - ys[i])
            return float(xs[i] + t * (xs[i + 1] - xs[i]))
    return None


def first_reach(sites, vals, target):
    for i in range(len(vals)):
        if vals[i] >= target:
            if i == 0:
                return float(sites[0])
            v0, v1 = vals[i - 1], vals[i]
            if v1 == v0:
                return float(sites[i])
            t = (target - v0) / (v1 - v0)
            return float(sites[i - 1] + t * (sites[i] - sites[i - 1]))
    return None


def _recover_of(rows, alpha_val=1.0):
    for r in rows:
        if abs(r['alpha'] - alpha_val) < 1e-12:
            return float(r['dDonor'] / FULL_SWAP)
    return float('nan')


ALPHAS = [r['alpha'] for r in E2[str(SWAP_SITES[0])]]
YMAT = {s: np.array([r['dDonor'] / FULL_SWAP for r in E2[str(s)]], float) for s in SWAP_SITES}
RECOVER = {s: _recover_of(E2[str(s)]) for s in SWAP_SITES}
RECOVER_R = _recover_of(E6)
MIN_REC = min(RECOVER[s] for s in SWAP_SITES)
SPAN = RECOVER[SWAP_SITES[-1]] - RECOVER[SWAP_SITES[0]]
XH = {s: cross_alpha(ALPHAS, YMAT[s], 0.5) for s in SWAP_SITES}
XH_SITES = [s for s in SWAP_SITES if XH[s] is not None]
XH_VALS = [XH[s] for s in XH_SITES]
XH_RANGE = (max(XH_VALS) - min(XH_VALS)) if len(XH_VALS) >= 4 else None
XSTAR = {s: (prof_swap[str(s)]['x_star']) for s in SWAP_SITES}
RHO_REC = spearman([RECOVER[s] for s in SWAP_SITES], SWAP_SITES) if len(SWAP_SITES) >= 4 else None
RHO_X = spearman(XH_VALS, XH_SITES) if len(XH_SITES) >= 4 else None
ALPHAS_R = [r['alpha'] for r in E6]
XH_R = cross_alpha(ALPHAS_R, np.array([r['dDonor'] / FULL_SWAP for r in E6], float), 0.5)
JS_OK = [s for s in SWAP_SITES
         if prof_swap[str(s)]['jump_ratio'] is not None and np.isfinite(prof_swap[str(s)]['jump_ratio'])
         and prof_swap[str(s)]['jump_ratio'] > 0 and prof_swap[str(s)]['y_sat'] >= CL['UNREACH_y']]
CURVE_OK_FRAC = len(JS_OK) / max(len(SWAP_SITES), 1)
G0_ok = bool(CURVE_OK_FRAC >= G['G0_precondition']['curve_ok_frac_min'] and
             XH_RANGE is not None and XH_RANGE >= G['G0_precondition']['xh_range_min'])

W = int(G['G2_concentration']['window'])
JUMPS_X = []; TOP3_X = None; MAXS_X = None
if len(XH_SITES) >= 4 and XH_RANGE is not None and XH_RANGE > 1e-9:
    JUMPS_X = [XH[XH_SITES[i + 1]] - XH[XH_SITES[i]] for i in range(len(XH_SITES) - 1)]
    WINS_X = [abs(sum(JUMPS_X[i:i + W])) for i in range(len(JUMPS_X) - W + 1)] if len(JUMPS_X) >= W else []
    TOP3_X = float(max(WINS_X) / XH_RANGE) if WINS_X else None
    MAXS_X = float(max(abs(j) for j in JUMPS_X) / XH_RANGE) if JUMPS_X else None

X_HALF = None; SPAN_10_90 = None
if XH_RANGE is None:
    G1 = 'G1_NA'
elif XH_RANGE < G['G0_precondition']['xh_range_min']:
    G1 = 'G1_flat'
elif len(XH_SITES) >= 4:
    lo = min(XH_VALS)
    XN = [(XH[s] - lo) / XH_RANGE for s in XH_SITES]
    X_HALF = first_reach(XH_SITES, XN, 0.5)
    D10 = first_reach(XH_SITES, XN, 0.1)
    D90 = first_reach(XH_SITES, XN, 0.9)
    SPAN_10_90 = (D90 - D10) if (D10 is not None and D90 is not None) else None
    if X_HALF is not None and SPAN_10_90 is not None and X_HALF <= 20 and SPAN_10_90 <= 14:
        G1 = 'G1a_crystallized'
    elif SPAN_10_90 is not None and SPAN_10_90 >= 24:
        G1 = 'G1b_distributed'
    else:
        G1 = 'G1_mid'
else:
    G1 = 'G1_NA'

if TOP3_X is None:
    G2 = 'G2_NA'
elif TOP3_X >= G['G2_concentration']['G2a_top3_share_x_min']:
    G2 = 'G2a_few_layer_dominant'
elif MAXS_X is not None and MAXS_X <= G['G2_concentration']['G2b_max_share_x_max']:
    G2 = 'G2b_layerwise_accumulate'
else:
    G2 = 'G2_mid'

_JG = [(s, prof_swap[str(s)]['jump_ratio'], J_INJECT.get(s)) for s in SWAP_SITES]
_JG = [(s, a, b) for (s, a, b) in _JG if a is not None and np.isfinite(a) and b is not None and np.isfinite(b)]
RHO_JG = spearman([x[1] for x in _JG], [x[2] for x in _JG]) if len(_JG) >= 4 else None
if RHO_JG is None:
    G3 = 'G3_NA'
elif RHO_JG >= G['G3_profile_shape']['rho_JG_same_min']:
    G3 = 'G3_same_gradient'
elif RHO_JG <= G['G3_profile_shape']['rho_JG_indep_max']:
    G3 = 'G3_independent'
else:
    G3 = 'G3_weak'

G5 = ('LAST_POS_STATE_SUFFICIENT' if MIN_REC >= G['G5_sufficiency']['min_recover_min']
      else 'LAST_POS_STATE_NOT_SUFFICIENT')


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
BRNG = np.random.default_rng(int(BOOT['seed']))


def _ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return dict(lo=None, hi=None, med=None, n_ok=int(len(v)))
    lo, hi = np.percentile(v, [2.5, 97.5])
    return dict(lo=float(lo), hi=float(hi), med=float(np.median(v)), n_ok=int(len(v)))


boot = dict(B=BS, recover_ci={}, J_ci={}, xhalf_ci={}, rho_recover=None, rho_xhalf=None,
            top3_share_x_ci=None, top3_share_recover_ci=None, R_ci={}, n_ok=0)
perm_x = np.array([]); perm_rec = np.array([]); PLO_X = PHI_X = PLO_R = PHI_R = None

if not SMOKE and len(SWAP_SITES) >= 4 and DISC_P:
    PM_swap = np.stack([np.array(E2P[str(s)], dtype=float) for s in SWAP_SITES], 0)
    PM_R = np.array(E6P, dtype=float)
    xs_sw = np.array(ALPHAS, float)
    nS, nA, nP = PM_swap.shape
    a1 = int(np.argmin(np.abs(xs_sw - 1.0)))
    rec_b = np.full((BS, nS), np.nan)
    J_b = np.full((BS, nS), np.nan)
    XH_b = np.full((BS, nS), np.nan)
    recR_b = np.full(BS, np.nan); xhR_b = np.full(BS, np.nan)
    rho_b = np.full(BS, np.nan); rhox_b = np.full(BS, np.nan)
    t3x_b = np.full(BS, np.nan); t3r_b = np.full(BS, np.nan)
    site_arr = np.array(SWAP_SITES, float)
    for b in range(BS):
        idx = BRNG.integers(0, nP, nP)
        fs_b = float(FS_VEC[idx].mean())
        if abs(fs_b) < 1e-9:
            continue
        Y = PM_swap[:, :, idx].mean(axis=2) / fs_b
        rec_b[b] = Y[:, a1]
        for i in range(nS):
            J_b[b, i] = J_only(xs_sw, Y[i])
            xv = cross_alpha(xs_sw, Y[i], 0.5)
            XH_b[b, i] = xv if xv is not None else np.nan
        rr = rec_b[b]
        if len(rr) >= 4 and np.all(np.isfinite(rr)):
            rho_b[b] = spearman(rr, site_arr)
            inc = np.diff(rr); tot = rr[-1] - rr[0]
            if tot > 1e-9 and len(inc) >= W:
                wins = [abs(sum(inc[i:i + W])) for i in range(len(inc) - W + 1)]
                t3r_b[b] = max(wins) / tot
        xr = XH_b[b]; okx = np.isfinite(xr)
        if okx.sum() >= 4:
            rhox_b[b] = spearman(xr[okx], site_arr[okx])
            rngx = float(xr[okx].max() - xr[okx].min())
            jm = np.diff(xr[okx])
            if rngx > 1e-9 and len(jm) >= W:
                wins = [abs(sum(jm[j:j + W])) for j in range(len(jm) - W + 1)]
                t3x_b[b] = max(wins) / rngx
        YR = PM_R[:, idx].mean(axis=1) / fs_b
        recR_b[b] = YR[a1]
        xrv = cross_alpha(xs_sw, YR, 0.5)
        xhR_b[b] = xrv if xrv is not None else np.nan
    boot['rho_recover'] = dict(hat=RHO_REC, **_ci(rho_b))
    boot['rho_xhalf'] = dict(hat=RHO_X, **_ci(rhox_b))
    boot['top3_share_x_ci'] = _ci(t3x_b)
    boot['top3_share_recover_ci'] = _ci(t3r_b)
    boot['R_ci'] = dict(recover=dict(hat=float(RECOVER_R), **_ci(recR_b)),
                        xhalf=dict(hat=XH_R, **_ci(xhR_b)))
    boot['n_ok'] = int(np.isfinite(rhox_b).sum())
    for i, s in enumerate(SWAP_SITES):
        boot['recover_ci'][str(s)] = dict(hat=float(RECOVER[s]), **_ci(rec_b[:, i]))
        boot['xhalf_ci'][str(s)] = dict(hat=(None if XH[s] is None else float(XH[s])), **_ci(XH_b[:, i]))
        boot['J_ci'][str(s)] = dict(
            hat=(float(prof_swap[str(s)]['jump_ratio'])
                 if prof_swap[str(s)]['jump_ratio'] is not None and np.isfinite(prof_swap[str(s)]['jump_ratio'])
                 else None), **_ci(J_b[:, i]))

if not SMOKE and len(XH_SITES) >= 4:
    perm_x = np.empty(BP); perm_rec = np.empty(BP)
    xv = np.array(XH_VALS, float)
    rv = np.array([RECOVER[s] for s in SWAP_SITES], float)
    xs_arr = np.array(XH_SITES, float)
    rs_arr = np.array(SWAP_SITES, float)
    for b in range(BP):
        perm_x[b] = spearman(BRNG.permutation(xv), xs_arr)
        perm_rec[b] = spearman(BRNG.permutation(rv), rs_arr)
    px = perm_x[np.isfinite(perm_x)]; pr = perm_rec[np.isfinite(perm_rec)]
    if len(px) >= 10:
        PLO_X, PHI_X = float(np.percentile(px, 2.5)), float(np.percentile(px, 97.5))
    if len(pr) >= 10:
        PLO_R, PHI_R = float(np.percentile(pr, 2.5)), float(np.percentile(pr, 97.5))
F7pp = bool(PLO_X is not None and PHI_X is not None and max(abs(PLO_X), abs(PHI_X)) < 0.6)

w('')
conf_out = {}
confP = {}
RHO_CONF = None; RHO_CONF_X = None; G4 = None
if CF_SITES and CONF_P:
    w('--- E5 确认集（n=%d，位点 %s）---' % (len(CONF_P), CF_SITES))
    G5A = grid_of('E5_conf_swap', 3)
    conf_rec = {}; conf_xh = {}
    for s in CF_SITES:
        rows = dose_swap(s, G5A, CONF_P, per_pair=True)
        confP[str(s)] = [x.get('per_pair', []) for x in rows]
        conf_rec[s] = _recover_of(rows)
        conf_xh[s] = cross_alpha([r['alpha'] for r in rows],
                                 np.array([r['dDonor'] / FULL_SWAP for r in rows], float), 0.5)
        d = curve_stats([r['alpha'] for r in rows], [r['dDonor'] / FULL_SWAP for r in rows], 'conf')
        conf_out[str(s)] = dict(rows=rows, cls=d['cls'], J=d['jump_ratio'], recover=float(conf_rec[s]),
                                xhalf=(None if conf_xh[s] is None else float(conf_xh[s])),
                                x_star=d['x_star'], y_sat=d['y_sat'])
        w('  L%-3s recover=%.4f  xhalf=%s  J=%s cls=%s' %
          (s, conf_rec[s], ('%.3f' % conf_xh[s]) if conf_xh[s] is not None else 'n/a',
           ('%.2f' % d['jump_ratio']) if d['jump_ratio'] is not None else 'n/a', d['cls']))
    RHO_CONF = spearman([conf_rec[s] for s in CF_SITES], CF_SITES) if len(CF_SITES) >= 4 else None
    _cx = [s for s in CF_SITES if conf_xh[s] is not None]
    RHO_CONF_X = spearman([conf_xh[s] for s in _cx], _cx) if len(_cx) >= 4 else None
    conf_out['rho_recover'] = RHO_CONF
    conf_out['rho_xhalf'] = RHO_CONF_X
    conf_out['min_recover'] = float(min(conf_rec[s] for s in CF_SITES))
    conf_out['G5'] = bool(conf_out['min_recover'] >= G['G5_sufficiency']['min_recover_min'])
    if not SMOKE and len(CF_SITES) >= 4:
        PMc = np.stack([np.array(confP[str(s)], dtype=float) for s in CF_SITES], 0)
        xsc = np.array([r['alpha'] for r in conf_out[str(CF_SITES[0])]['rows']], float)
        a1c = int(np.argmin(np.abs(xsc - 1.0)))
        nPc = PMc.shape[2]
        rb = np.full(BS, np.nan); xb = np.full(BS, np.nan)
        for b in range(BS):
            idx = BRNG.integers(0, nPc, nPc)
            Yc = PMc[:, :, idx].mean(axis=2) / FULL_SWAP
            rb[b] = spearman(Yc[:, a1c], CF_SITES)
            xr = [cross_alpha(xsc, Yc[k], 0.5) for k in range(len(CF_SITES))]
            xr = np.array([np.nan if v is None else v for v in xr], float)
            ok = np.isfinite(xr)
            if ok.sum() >= 4:
                xb[b] = spearman(xr[ok], np.array(CF_SITES, float)[ok])
        conf_out['rho_boot_recover'] = _ci(rb)
        conf_out['rho_boot_xhalf'] = _ci(xb)
    G4 = bool(RHO_CONF_X is not None and RHO_X is not None and
              (RHO_CONF_X * RHO_X > 0) and abs(RHO_CONF_X) >= G['G4_confirmation']['rho_abs_min'])
    conf_out['G4'] = G4
else:
    w('  (确认集在冒烟下跳过)')
sys.stdout.flush()

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

maxabs = max([abs(r['dDonor']) for s in E2 for r in E2[s]] +
             [abs(r['dDonor']) for s in E2b for r in E2b[s]] +
             [abs(r['dDonor']) for s in E3 for r in E3[s]] +
             [abs(r['dDonor']) for r in E6] + [1e-9])
E4max = max(abs(r['dDonor']) for r in E4)
F1_ok = bool(E4max < 0.10 * maxabs)
offm = sorted(set([round(r['alpha'], 4) for s in E2 for r in E2[s] if (r['pert_rel'] or 0) > PERT_LIM]))
F10_ok = True; F10_max = 0.0
for s in SWAP_SITES:
    for row, per in zip(E2[str(s)], E2P[str(s)]):
        if per:
            dv = abs(float(np.mean(per)) - row['dDonor'])
            F10_max = max(F10_max, dv)
            if dv > 1e-9:
                F10_ok = False
F12_ok = bool(all(v < 1e-5 for v in f12.values()))
el = time.time() - t0

w('')
w('=== 主表：xhalf / J_swap / recover 剖面 ===')
w('%-6s %9s %9s %9s %9s %9s %-13s %7s %7s' %
  ('site', 'xhalf', 'x*_log', 'recover', 'J_swap', 'J_inject', 'cls_swap', 'q_ell', 'psu_u6'))
for s in SWAP_SITES:
    jf = J_INJECT.get(s)
    jsw = prof_swap[str(s)]['jump_ratio']
    w('L%-5d %9s %9s %9.4f %9s %9s %-13s %7.3f %7.3f' % (
        s,
        ('%.3f' % XH[s]) if XH[s] is not None else 'n/a',
        ('%.3f' % XSTAR[s]) if XSTAR[s] is not None else 'n/a',
        RECOVER[s],
        ('%.2f' % jsw) if jsw is not None and np.isfinite(jsw) else 'n/a',
        ('%.2f' % jf) if jf is not None else 'n/a',
        prof_swap[str(s)]['cls'], Q_ELL[str(s)], PSU[str(s)]))
w('R      %9s %9s %9.4f %9s %9s %-13s %7.3f %7.3f' % (
    ('%.3f' % XH_R) if XH_R is not None else 'n/a',
    ('%.3f' % prof_R_swap['x_star']) if prof_R_swap['x_star'] is not None else 'n/a',
    RECOVER_R,
    ('%.2f' % prof_R_swap['jump_ratio']) if prof_R_swap['jump_ratio'] is not None and np.isfinite(prof_R_swap['jump_ratio']) else 'n/a',
    'n/a', prof_R_swap['cls'], Q_R, PSU_R))
w('')
w('=== G 族判决（amend1 口径）===')
w('  G0 curve_ok_frac = %d/%d = %.3f (阈值 %.2f) ; XH_RANGE = %s (阈值 %.2f) -> ok=%s' % (
    len(JS_OK), len(SWAP_SITES), CURVE_OK_FRAC, G['G0_precondition']['curve_ok_frac_min'],
    ('%.4f' % XH_RANGE) if XH_RANGE is not None else 'n/a', G['G0_precondition']['xh_range_min'], G0_ok))
w('  rho(xhalf, depth) = %s (带 %s)' % (
    ('%.4f' % RHO_X) if RHO_X is not None else 'n/a',
    ('[%.4f, %.4f]' % ((boot['rho_xhalf'] or {}).get('lo'), (boot['rho_xhalf'] or {}).get('hi')))
    if (boot['rho_xhalf'] or {}).get('lo') is not None else 'n/a'))
w('  G1 x_half = %s ; span_10_90 = %s -> %s' % (
    ('%.2f' % X_HALF) if X_HALF is not None else 'n/a',
    ('%.2f' % SPAN_10_90) if SPAN_10_90 is not None else 'n/a', G1))
w('  G2 top3_share_x = %s (带 %s) ; max_share_x = %s -> %s' % (
    ('%.4f' % TOP3_X) if TOP3_X is not None else 'n/a',
    ('[%.4f, %.4f]' % ((boot['top3_share_x_ci'] or {}).get('lo'), (boot['top3_share_x_ci'] or {}).get('hi')))
    if (boot['top3_share_x_ci'] or {}).get('lo') is not None else 'n/a',
    ('%.4f' % MAXS_X) if MAXS_X is not None else 'n/a', G2))
w('  G3 rho(J_swap, J_inject) = %s (n=%d) -> %s' % (
    ('%.4f' % RHO_JG) if RHO_JG is not None else 'n/a', len(_JG), G3))
w('  G4 确认集 rho(xhalf) = %s -> %s' % (('%.4f' % RHO_CONF_X) if RHO_CONF_X is not None else 'n/a', G4))
w('  G5 min recover = %.4f (阈值 %.2f) -> %s' % (MIN_REC, G['G5_sufficiency']['min_recover_min'], G5))
w('  置换零假设(xhalf) 95%% 带 = [%s, %s] (|界|<0.6?) -> %s' % (
    ('%.4f' % PLO_X) if PLO_X is not None else 'n/a',
    ('%.4f' % PHI_X) if PHI_X is not None else 'n/a', F7pp))
w('  ==> G 族裁决 = %s' % G_VERDICT)
w('')
w('=== 诊断：recover 端点量（amend1 A1：按构造饱和）===')
w('  span(recover) = %+.4f ; min = %.4f ; rho(recover, depth) = %s' % (
    SPAN, MIN_REC, ('%.4f' % RHO_REC) if RHO_REC is not None else 'n/a'))
w('  recover 剖面：%s' % '  '.join('L%d:%.4f' % (s, RECOVER[s]) for s in SWAP_SITES))
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
    E2_pairs=E2P, E2b_pairs=E2bP, E5_pairs={k: v for k, v in confP.items()}, E6_pairs=E6P,
    profile_swap=prof_swap, profile_swap_rel=prof_swap_rel, profile_R_swap=prof_R_swap,
    recover=dict(curve={str(s): float(RECOVER[s]) for s in SWAP_SITES},
                 R=float(RECOVER_R), span=float(SPAN), min=float(MIN_REC), rho=RHO_REC, G5=G5),
    xhalf=dict(curve={str(s): (None if XH[s] is None else float(XH[s])) for s in SWAP_SITES},
               R=(None if XH_R is None else float(XH_R)),
               x_star_logistic={str(s): XSTAR[s] for s in SWAP_SITES},
               x_star_R=prof_R_swap['x_star'],
               sites=XH_SITES, range=XH_RANGE, rho=RHO_X, jumps=JUMPS_X,
               top3_share_x=TOP3_X, max_share_x=MAXS_X, window=W),
    G_family=dict(
        G0=dict(ok=bool(G0_ok), curve_ok_frac=CURVE_OK_FRAC, n_curve_ok=len(JS_OK),
                xh_range=XH_RANGE, curve_ok_frac_min=G['G0_precondition']['curve_ok_frac_min'],
                xh_range_min=G['G0_precondition']['xh_range_min']),
        G1=dict(label=G1, x_half=X_HALF, span_10_90=SPAN_10_90),
        G2=dict(label=G2, top3_share_x=TOP3_X, max_share_x=MAXS_X,
                top3_share_x_min=G['G2_concentration']['G2a_top3_share_x_min'],
                max_share_x_max=G['G2_concentration']['G2b_max_share_x_max']),
        G3=dict(label=G3, rho_JG=RHO_JG, n_sites=len(_JG), pairs=[(s, a, b) for (s, a, b) in _JG]),
        G4=dict(label=('G4_ok' if G4 else ('G4_fail' if G4 is not None else 'G4_na')),
                rho_conf_xhalf=RHO_CONF_X, rho_conf_recover=RHO_CONF),
        G5=dict(label=G5, min_recover=float(MIN_REC),
                min_recover_min=G['G5_sufficiency']['min_recover_min'],
                passed=bool(MIN_REC >= G['G5_sufficiency']['min_recover_min'])),
        F7pp=bool(F7pp),
    ),
    G_verdict=G_VERDICT,
    bootstrap_band=boot,
    permutation_null=dict(B=BP, xhalf=dict(lo=PLO_X, hi=PHI_X, values=perm_x.tolist()),
                          recover=dict(lo=PLO_R, hi=PHI_R, values=perm_rec.tolist())),
    floors=dict(F1_ok=bool(F1_ok), E4_max=E4max, maxabs_all=maxabs,
                F3_dev=f3, F3_ok=bool(all(v < 1e-2 for v in f3.values())),
                F6_ok=bool(E0['dDonor'] == FULL_REF),
                F10_ok=bool(F10_ok), F10_max=F10_max,
                F11_ok=bool(F11_ok), F11_max=F11_max, F11_val=F11_val,
                F12_ok=bool(F12_ok), F12_dev=f12,
                F13_ok=bool(sha(R11P) == E['phase11_result_sha256'])),
    off_manifold_alphas=offm,
    drift_flags=dict(n6_drift=bool(N6_DRIFT), off_manifold=offm),
    seal_sha8=sha(SEAL)[:8], amend1_sha8=sha(AMEND)[:8], exec_sha8=sha(EXEC)[:8],
    inherits_panel_sha8=E['inherits_panel_sha256'][:8],
    phase11_result_sha8=E['phase11_result_sha256'][:8],
)
io.open(RESULT, 'w', encoding='utf-8').write(json.dumps(res, ensure_ascii=False, indent=1))
io.open(REPORT, 'w', encoding='utf-8').write('\n'.join(lines))
print('DONE ->', REPORT, '|', RESULT)
