# -*- coding: utf-8 -*-
"""
Phase 9 / N2h1-alpha-2 : 承诺层「阈值增益」的剂量-响应直接检验
=========================================================================
预注册：tests/deepseek_temp/Phase9/N2h1a2_design_seal.json（观测前冻结）
执行冻结：tests/deepseek_temp/Phase9/execution_phase9.json（面板逐字节继承 Phase 8 + 臂/alpha 网格）

要否证的命题（Phase 8 §7）：
  「L6 不是把类别轴"写进去"，而是把上游已有的分量（23.2% 范数、2.7% 效应）放大到锁定量级；
    该放大是阈值型的。」

设计（两站点 × 剂量曲线）：
  D1 站点 S_L6out : h6r + a * P_U6(diff6)         -> 「写入 -> 行为」传递函数
  D2 站点 S_L5out : h5r + a * P_U6(diff5)         -> 「上游 -> 行为」+ L6 自增益 amp(a)
  D1a 锚点 / D2b 基变换 / D3 正交补 / D4 随机 5 维 / D6 上游正交补
判据：H1 亚阈值突变 / H2 连续超线性(幂律) / H3 线性中继 / H4 不可达（优先级 H4>H1>H2>H3）
用法：SMOKE=1 python n2h1a2_threshold_gain.py   # 冒烟（面板截断）
      python n2h1a2_threshold_gain.py           # 正式
"""
import os, sys, io, json, time, hashlib
import numpy as np
import torch

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P9T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase9')
EXEC = os.path.join(P9T, 'execution_phase9.json')
SEAL = os.path.join(P9T, 'N2h1a2_design_seal.json')
REPORT = os.path.join(P9T, 'n2h1a2_report_qwen3-4b.txt')
RESULT = os.path.join(P9T, 'result_phase9.json')
SMOKE = os.environ.get('SMOKE', '0') == '1'
if SMOKE:
    # 冒烟产物一律落 smoke/ 子目录，绝不覆写正式产物（Phase 8 教训）
    _S = os.path.join(P9T, 'smoke')
    os.makedirs(_S, exist_ok=True)
    REPORT = os.path.join(_S, 'n2h1a2_report_qwen3-4b.txt')
    RESULT = os.path.join(_S, 'result_phase9.json')

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
PRIMARY = E['primary_layer']          # 6
PRE = E['pre_layer']                  # 5
NH, HD = E['n_heads'], E['head_dim']
RNG = np.random.default_rng(E['seed'])
ARMS = E['arms']
RBAR_REF = float(E['rbar_ref_from_phase8'])
RBAR_TOL = float(E['rbar_drift_tol'])
ANCHOR_REF = float(E['anchor_ref_dDonor_phase8_diff5'])
ANCHOR_TOL = float(E['anchor_drift_tol'])
PERT_LIM = float(E['off_manifold_pert_rel'])


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


w('=== Phase 9 / N2h1-alpha-2 : 承诺层阈值增益的剂量-响应直接检验 ===')
w('smoke=%s ; time %s' % (SMOKE, time.strftime('%Y-%m-%d %H:%M:%S')))
w('seal sha8 %s ; exec sha8 %s' % (sha(SEAL)[:8], sha(EXEC)[:8]))
w('exec inherits panel from %s (sha256 %s)' % (E['inherits_panel_from'], E['inherits_panel_sha256'][:16]))
w('config_sha256_match %s (expect %s)' % (
    sha(os.path.join(MDIR, 'config.json')) == E['config_sha256'], E['config_sha256'][:12]))

from transformers import AutoTokenizer, AutoModelForCausalLM
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MDIR, dtype=torch.bfloat16, trust_remote_code=True,
                                            attn_implementation='eager').to('cuda').eval()
_core = getattr(model.model, 'language_model', model.model)
layers = _core.layers; L = len(layers); head_lm = model.lm_head
HID = model.config.hidden_size
CFG = model.config

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

# ---------------- 0. drift 断言 ----------------
drift = []
if L != E['expected_cfg']['num_hidden_layers']: drift.append('layers')
if HID != E['expected_cfg']['hidden_size']: drift.append('hidden')
if CFG.num_attention_heads != NH: drift.append('n_heads')
if bool(CFG.tie_word_embeddings) != bool(E['expected_cfg']['tie_word_embeddings']): drift.append('tie')
if OIN != E['o_proj_in_features']: drift.append('o_proj_in %d' % OIN)
w('drift(F4): %s' % (drift if drift else 'NONE'))
assert OIN == NH * HD, 'F4 失败：o_proj 输入维 %d != %d' % (OIN, NH * HD)
assert L > PRIMARY + 1
if drift:
    w('!! DRIFT 非空，按预注册 F4 停止'); sys.exit(2)

PAIRS_ALL = [tuple(p) for p in E['pairs_all']]
DISC = [tuple(x) for x in E['discovery']]
CONF = [tuple(x) for x in E['confirmation']]
INST_ALL = [tuple(x) for x in E['instances_all']]

def grid_of(arm, n_smoke=3):
    g = ARMS[arm]['alpha_grid']
    if isinstance(g, str):
        return g                       # 'rbar' 单点，运行时才定
    if not SMOKE:
        return list(g)
    sub = list(g[:n_smoke])
    if 1.0 in g and 1.0 not in sub:    # 冒烟也保留 alpha=1（full 的定义点）
        sub.append(1.0)
    return sub

if SMOKE:
    DISC = DISC[:2]
    CONF = []
    _need = set()
    for p in PAIRS_ALL:
        if p[0] in [x[0] for x in DISC]:
            _need.add(p[0]); _need.add(p[2])
    INST_ALL = [(a, b) for (a, b) in INST_ALL if a in _need]
    w('SMOKE panel: discovery=%d ; captured=%s' % (len(DISC), [x[0] for x in INST_ALL]))

w('')
w('model=%s L=%d hid=%d heads=%d head_dim=%d o_proj_in=%d tie=%s' %
  (MODEL, L, HID, NH, HD, OIN, CFG.tie_word_embeddings))
w('site S_L6out=layers[%d] ; S_L5out=layers[%d] ; template=%r ; seed=%d' % (PRIMARY, PRE, TMPL, E['seed']))
w('panel: discovery=%d confirmation=%d (all=%d)' % (len(DISC), len(CONF), len(INST_ALL)))
sys.stdout.flush()

# ---------------- 1. 采集（只需 hidden states）----------------
@torch.no_grad()
def capture(text):
    ii = torch.tensor([ids_of(text)], device='cuda')
    out = model(input_ids=ii, output_hidden_states=True)
    HH = np.stack([h[0, -1].float().detach().cpu().numpy() for h in out.hidden_states], 0)
    return HH, out.logits[0, -1].float().detach().cpu().numpy()

t_cap = time.time()
CAP = {}
for wd, sup in INST_ALL:
    CAP[wd] = capture(TMPL % wd)
w('capture done %d instances in %.1fs' % (len(CAP), time.time() - t_cap))
if SMOKE:
    wd0 = INST_ALL[0][0]
    HH, lg = CAP[wd0]
    w('SMOKE assert: HH%s logits%s nan=%s' % (HH.shape, lg.shape, bool(np.isnan(HH).any())))
    assert HH.shape[1] == HID and not np.isnan(HH).any()
PAIRS = [p for p in PAIRS_ALL if p[0] in CAP and p[2] in CAP]
w('usable pairs (recipient & donor captured) = %d / %d' % (len(PAIRS), len(PAIRS_ALL)))
sys.stdout.flush()

H5 = PRE + 1          # hidden_states[6] = 层的 5 输出 = L6 的输入
H6 = PRIMARY + 1      # hidden_states[7] = L6 输出

# ---------------- 2. 类子空间 U5 / U6（只用 discovery 估计）----------------
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

U5, SV5, nA5 = est_U(H5, DISC)
U6, SV6, nA6 = est_U(H6, DISC)
U6 = U6 if nA6 >= 4 else est_U(H6, INST_ALL)[0]
U5 = U5 if nA5 >= 4 else est_U(H5, INST_ALL)[0]
AO = U6
w('U estimated on discovery only: n_classes=%d ; U5%s ; U6%s' % (nA5, U5.shape, U6.shape))
w('  sing@U6 = %s' % ' '.join('%.1f' % x for x in SV6))
sys.stdout.flush()

# 主角诊断：U5 与 U6 的子空间重叠
Mcross = U5 @ U6.T
sv_c = np.linalg.svd(Mcross, compute_uv=False)
overlap = float(np.sum(sv_c ** 2) / U6.shape[0])
w('U5/U6 principal cos = %s ; overlap(sum cos^2/rank) = %.4f' %
  (' '.join('%.3f' % x for x in sv_c), overlap))
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
    lg = CAP[rw][1]
    BASE[rw] = dict(sr0=score_of(lg, rs, ids_of(rw)[0]),
                    sd0=score_of(lg, ds, ids_of(dw)[0]),
                    rd0=rank_of(lg, ds, ids_of(dw)[0]))
bad_base = [rw for rw, b in BASE.items() if not (b['sr0'] > 0)]
base_r1 = float(np.mean([1.0 if BASE[rw]['rd0'] == 1 else 0.0 for rw in BASE]))
w('base: n=%d ; 受体类分数 mean=%+.3f ; 供体类分数 mean=%+.3f ; 供体类已 rank1 比例=%.3f ; base<=0 = %s (F2)' %
  (len(BASE), np.mean([b['sr0'] for b in BASE.values()]), np.mean([b['sd0'] for b in BASE.values()]),
   base_r1, bad_base if bad_base else 'NONE'))
assert not bad_base, 'F2 失败'
sys.stdout.flush()

# ---------------- 4. 每对的写入向量与上游分量 ----------------
VEC = {}
for (rw, rs, dw, ds, sw) in PAIRS:
    h5r, h5d = CAP[rw][0][H5].astype(np.float32), CAP[dw][0][H5].astype(np.float32)
    h6r, h6d = CAP[rw][0][H6].astype(np.float32), CAP[dw][0][H6].astype(np.float32)
    d5, d6 = h5d - h5r, h6d - h6r
    u5, u6 = proj(d5, AO), proj(d6, AO)
    u5b = proj(d5, U5)
    VEC[rw] = dict(h5r=h5r, h6r=h6r, d5=d5, d6=d6, u5=u5, u6=u6, u5b=u5b,
                   n5=float(np.linalg.norm(u5)), n6=float(np.linalg.norm(u6)),
                   n5b=float(np.linalg.norm(u5b)),
                   nh5=float(np.linalg.norm(h5r)), nh6=float(np.linalg.norm(h6r)),
                   nd6=float(np.linalg.norm(d6)),
                   comp6=d6 - u6, comp5=d5 - u5)
_n5 = np.mean([VEC[rw]['n5'] for rw in VEC])
_n6 = np.mean([VEC[rw]['n6'] for rw in VEC])
RBAR = float(_n5 / _n6)
RPER = np.array([VEC[rw]['n5'] / max(VEC[rw]['n6'], 1e-9) for rw in VEC])
RBAR_DRIFT = abs(RBAR - RBAR_REF) > RBAR_TOL
w('')
w('--- 剂量坐标 ---')
w('mean||P_U6(diff6)||=%.3f ; mean||P_U6(diff5)||=%.3f ; rbar=%.5f (Phase8 参照 %.5f, drift=%s)' %
  (_n6, _n5, RBAR, RBAR_REF, RBAR_DRIFT))
w('per-pair r: mean=%.4f sd=%.4f min=%.4f max=%.4f' % (RPER.mean(), RPER.std(), RPER.min(), RPER.max()))
w('mean||P_U5(diff5)||=%.3f ; mean||diff6||=%.3f ; mean||h5_recip||=%.1f ; 1/rbar=%.3f' %
  (np.mean([VEC[rw]['n5b'] for rw in VEC]), np.mean([VEC[rw]['nd6'] for rw in VEC]),
   np.mean([VEC[rw]['nh5'] for rw in VEC]), 1.0 / RBAR))
sys.stdout.flush()

# ---------------- 5. 前向工具 ----------------
@torch.no_grad()
def fwd_patch(text, site, vec, want_h6=False):
    """替换 layers[site] 末位输出为 vec；可选返回 L6 输出末位（用于 amp）"""
    ii = torch.tensor([ids_of(text)], device='cuda')
    def hook(mod, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        t = t.clone(); t[0, -1, :] = vec.to(t.dtype)
        return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t
    h = layers[site].register_forward_hook(hook)
    o = model(input_ids=ii) if not want_h6 else model(input_ids=ii, output_hidden_states=True)
    h.remove()
    lg = o.logits[0, -1].float().detach().cpu().numpy()
    if want_h6:
        return lg, o.hidden_states[H6][0, -1].float().detach().cpu().numpy()
    return lg


def unit_in_U(Ub):
    z = RNG.standard_normal(Ub.shape[0]).astype(np.float32)
    v = z @ Ub
    nv = np.linalg.norm(v)
    return v / max(nv, 1e-9)


# ---------------- 6. F3：alpha=0 恒等自检（钩子正确性）----------------
w('')
w('--- F3 自检：alpha=0 注入必须与未干预前向同分 ---')
f3 = {}
for tag, site, key in [('L6out', PRIMARY, 'h6r'), ('L5out', PRE, 'h5r')]:
    devs = []
    for (rw, rs, dw, ds, sw) in PAIRS[:3]:
        lg0 = fwd_patch(TMPL % rw, site, torch.tensor(VEC[rw][key], device='cuda'))
        devs.append(abs(score_of(lg0, rs, ids_of(rw)[0]) - BASE[rw]['sr0']))
    f3[tag] = float(max(devs))
w('  max|dScore(alpha=0)| : L6out=%.3e  L5out=%.3e  (F3 要求 == 0)' % (f3['L6out'], f3['L5out']))
assert f3['L6out'] < 1e-2 and f3['L5out'] < 1e-2, 'F3 失败：钩子未还原基线'
sys.stdout.flush()

# ---------------- 7. 剂量臂通用执行器 ----------------
def run_dose(site, alpha_list, vec_fn, pairs, want_amp=False, rbar=RBAR):
    """vec_fn(rw, a) -> 注入向量；返回逐 alpha 统计"""
    out = []
    for a in alpha_list:
        dd, dr, r1, n = 0.0, 0.0, 0, 0
        amps, perts = [], []
        for (rw, rs, dw, ds, sw) in pairs:
            if rw not in VEC:
                continue
            B = BASE[rw]; sid_r = ids_of(rw)[0]; sid_d = ids_of(dw)[0]
            V = VEC[rw]
            vec = vec_fn(rw, a)
            if want_amp:
                lg, h6a = fwd_patch(TMPL % rw, site, torch.tensor(vec, device='cuda'), want_h6=True)
                dU = proj(h6a.astype(np.float32) - V['h6r'], AO)
                if abs(a) > 1e-9:
                    amps.append(float(np.linalg.norm(dU)) / (a * V['n5']))
                perts.append(a * V['n5'] / max(V['nh5'], 1e-9))
            else:
                lg = fwd_patch(TMPL % rw, site, torch.tensor(vec, device='cuda'))
            dd += score_of(lg, ds, sid_d) - B['sd0']
            dr += score_of(lg, rs, sid_r) - B['sr0']
            r1 += 1 if rank_of(lg, ds, sid_d) == 1 else 0
            n += 1
        out.append(dict(alpha=float(a), dDonor=dd / max(n, 1), dRecip=dr / max(n, 1),
                        rank1=r1 / max(n, 1), n=n,
                        amp=float(np.mean(amps)) if amps else None,
                        pert_rel=float(np.mean(perts)) if perts else None))
    return out


DISC_P = [p for p in PAIRS if p[0] in [x[0] for x in DISC]]

w('')
w('--- D1 站点 S_L6out ：h6r + a*P_U6(diff6)（discovery n=%d）---' % len(DISC_P))
gD1 = grid_of('D1_write_readout')
D1 = run_dose(PRIMARY, gD1, lambda rw, a: VEC[rw]['h6r'] + a * VEC[rw]['u6'], DISC_P)
for r in D1:
    w('  a=%6.3f  dDonor=%+8.3f  dRecip=%+8.3f  rank1=%.3f' % (r['alpha'], r['dDonor'], r['dRecip'], r['rank1']))
sys.stdout.flush()

w('')
w('--- D1a 锚点：a=rbar=%.5f（应对齐 Phase 8 diff5 臂 %+.3f）---' % (RBAR, ANCHOR_REF))
D1a = run_dose(PRIMARY, [RBAR], lambda rw, a: VEC[rw]['h6r'] + a * VEC[rw]['u6'], DISC_P)[0]
ANCHOR_DRIFT = abs(D1a['dDonor'] - ANCHOR_REF) > ANCHOR_TOL
w('  dDonor=%+8.3f  (ref %+.3f, |Δ|=%.3f, tol=%.2f, drift=%s)' %
  (D1a['dDonor'], ANCHOR_REF, abs(D1a['dDonor'] - ANCHOR_REF), ANCHOR_TOL, ANCHOR_DRIFT))
sys.stdout.flush()

w('')
w('--- D2 站点 S_L5out ：h5r + a*P_U6(diff5)（主臂，含 L6 自增益 amp）---')
gD2 = grid_of('D2_upstream')
D2 = run_dose(PRE, gD2, lambda rw, a: VEC[rw]['h5r'] + a * VEC[rw]['u5'], DISC_P, want_amp=True)
for r in D2:
    w('  a=%6.3f  x=%6.4f  dDonor=%+8.3f  dRecip=%+8.3f  rank1=%.3f  amp=%s  pert_rel=%s' %
      (r['alpha'], r['alpha'] * RBAR, r['dDonor'], r['dRecip'], r['rank1'],
       ('%.3f' % r['amp']) if r['amp'] is not None else '  -  ',
       ('%.3f' % r['pert_rel']) if r['pert_rel'] is not None else '  -  '))
sys.stdout.flush()

w('')
w('--- D2b 基变换稳健性：P_U5 基 ---')
gD2b = grid_of('D2b_basis_swap')
D2b = run_dose(PRE, gD2b, lambda rw, a: VEC[rw]['h5r'] + a * VEC[rw]['u5b'], DISC_P)
for r in D2b:
    w('  a=%6.3f  dDonor=%+8.3f' % (r['alpha'], r['dDonor']))
sys.stdout.flush()

w('')
w('--- D3 正交补特异性（S_L6out）---')
gD3 = grid_of('D3_orth_complement')
D3 = run_dose(PRIMARY, gD3, lambda rw, a: VEC[rw]['h6r'] + a * VEC[rw]['comp6'], DISC_P)
for r in D3:
    w('  a=%6.3f  dDonor=%+8.3f' % (r['alpha'], r['dDonor']))
sys.stdout.flush()

w('')
w('--- D4 随机 5 维（U6 内，范数匹配，3 draws）---')
gD4 = grid_of('D4_random5')
_draws = ARMS['D4_random5']['draws']
D4 = []
for a in gD4:
    dd, n = 0.0, 0
    for (rw, rs, dw, ds, sw) in DISC_P:
        B = BASE[rw]; sid_d = ids_of(dw)[0]; V = VEC[rw]
        for _ in range(_draws):
            v = unit_in_U(AO) * (a * V['n6'])
            lg = fwd_patch(TMPL % rw, PRIMARY, torch.tensor(V['h6r'] + v, device='cuda'))
            dd += score_of(lg, ds, sid_d) - B['sd0']; n += 1
    D4.append(dict(alpha=float(a), dDonor=dd / max(n, 1), n=n))
    w('  a=%6.3f  dDonor=%+8.3f  (n=%d)' % (a, D4[-1]['dDonor'], n))
sys.stdout.flush()

w('')
w('--- D6 上游正交补对照（S_L5out）---')
gD6 = grid_of('D6_upstream_orth')
D6 = run_dose(PRE, gD6, lambda rw, a: VEC[rw]['h5r'] + a * VEC[rw]['comp5'], DISC_P)
for r in D6:
    w('  a=%6.3f  dDonor=%+8.3f' % (r['alpha'], r['dDonor']))
sys.stdout.flush()

w('')
w('--- D7 上游 U 分量撤除的必要性（S_L5out，文本 = 供体句）[amend1] ---')
DB = {}
for (rw, rs, dw, ds, sw) in PAIRS:
    if dw in CAP:
        lg = CAP[dw][1]; sid_d = ids_of(dw)[0]
        DB[dw] = dict(sD1=score_of(lg, ds, sid_d), rD1=rank_of(lg, ds, sid_d))
_D7P = [p for p in DISC_P if p[2] in DB]
REF_D7 = float(np.mean([BASE[p[0]]['sd0'] for p in _D7P]) - np.mean([DB[p[2]]['sD1'] for p in _D7P]))
w('  供体基线: sD1 mean=%+.3f ; 供体句 donor-class rank1 比例=%.3f ; REF(受体->供体落差)=%+.3f' %
  (np.mean([DB[p[2]]['sD1'] for p in _D7P]),
   np.mean([1.0 if DB[p[2]]['rD1'] == 1 else 0.0 for p in _D7P]), REF_D7))
gD7 = grid_of('D7_l5_u_necessity')
D7 = []
for a in gD7:
    acc, r1, n = 0.0, 0, 0
    for (rw, rs, dw, ds, sw) in _D7P:
        V = VEC[rw]; sid_d = ids_of(dw)[0]
        vec = CAP[dw][0][H5].astype(np.float32) - (1.0 - a) * V['u5']
        lg = fwd_patch(TMPL % dw, PRE, torch.tensor(vec, device='cuda'))
        acc += score_of(lg, ds, sid_d) - DB[dw]['sD1']
        r1 += 1 if rank_of(lg, ds, sid_d) == 1 else 0
        n += 1
    dD = acc / max(n, 1)
    D7.append(dict(alpha=float(a), dD=dD, rank1=r1 / max(n, 1), n=n,
                   kill_frac=(abs(dD) / abs(REF_D7)) if abs(REF_D7) > 1e-9 else None))
    w('  a=%6.3f  dD=%+8.3f  donor_rank1=%.3f  kill_frac=%s' %
      (a, dD, D7[-1]['rank1'],
       ('%.3f' % D7[-1]['kill_frac']) if D7[-1]['kill_frac'] is not None else 'n/a'))
sys.stdout.flush()

# ---------------- 8. 判据 ----------------
FULL = float('nan')
_aD1 = [r['alpha'] for r in D1]
if 1.0 in _aD1:
    FULL = D1[_aD1.index(1.0)]['dDonor']
else:
    FULL = D1[-1]['dDonor']
    w('!! D1 网格不含 alpha=1，full 退回网格末点 alpha=%.3f' % D1[-1]['alpha'])
assert abs(FULL) > 1e-6, 'full 退化，无法归一化'
maxabs = max([abs(r['dDonor']) for r in D1 + D2 + D3 + D4 + D6] + [1e-9])


def curve_class(dose, rbar=RBAR, tag=''):
    """返回 (class, 细节dict)；dose = run_dose 的输出列表"""
    xs = np.array([r['alpha'] * rbar for r in dose], dtype=float)
    ys = np.array([r['dDonor'] / FULL for r in dose], dtype=float)
    m = xs >= 0.01
    xs2, ys2 = xs[m], ys[m]
    s = np.diff(ys2) / np.diff(xs2)
    if len(s) >= 2:
        k = int(np.argmax(s)); s_med = float(np.median(np.delete(s, k)))
        J = float(s[k] / s_med) if s_med > 0 else float('inf')
    else:
        k, s_med, J = -1, float('nan'), float('nan')
    mf = xs >= 0.10
    xf, yf = xs[mf], ys[mf]
    det = dict(x=xs.tolist(), y=ys.tolist(), slopes=s.tolist(), jump_ratio=J,
               argmax_slope_k=k, median_other_slope=s_med,
               x_max=float(xs.max()), y_max=float(ys.max()), y_at_x_max=float(ys[-1]))
    # 线性拟合
    if len(xf) >= 3:
        b1, b0 = np.polyfit(xf, yf, 1)
        pred = b0 + b1 * xf
        ss = 1.0 - np.sum((yf - pred) ** 2) / max(np.sum((yf - yf.mean()) ** 2), 1e-12)
        det['lin'] = dict(slope=float(b1), intercept=float(b0), R2=float(ss))
    else:
        det['lin'] = dict(slope=None, intercept=None, R2=None)
    # 幂律拟合 y = c x^g（需 y>0）
    if len(xf) >= 3 and np.all(yf > 0):
        g, lc = np.polyfit(np.log(xf), np.log(yf), 1)
        pred = lc + g * np.log(xf)
        ss = 1.0 - np.sum((np.log(yf) - pred) ** 2) / max(np.sum((np.log(yf) - np.log(yf).mean()) ** 2), 1e-12)
        det['pow'] = dict(gamma=float(g), c=float(np.exp(lc)), R2=float(ss))
    else:
        det['pow'] = dict(gamma=None, c=None, R2=None)
    # H1：跳变区间
    h1_ok, x_star = False, None
    if len(s) >= 2 and np.isfinite(J):
        left = float(xs2[k]); right = float(xs2[k + 1])
        y_left_max = float(np.max(ys2[:k + 1]))
        h1_ok = bool(J >= 5 and s_med > 0 and float(ys2[k + 1]) >= 0.50 and y_left_max <= 0.15)
        x_star = 0.5 * (left + right)
    det['x_star'] = x_star
    # 分类
    if det['x_max'] > 0 and float(np.max(np.abs(ys))) < 0.15 and tag == 'D2':
        cls = 'H4_unreachable'
    elif h1_ok:
        cls = 'H1_threshold'
    elif det['pow']['R2'] is not None and det['pow']['R2'] >= 0.98 and det['pow']['gamma'] >= 2:
        cls = 'H2_powerlaw'
    elif det['lin']['R2'] is not None and det['lin']['R2'] >= 0.97 and \
            det['pow']['gamma'] is not None and det['pow']['gamma'] <= 1.5:
        cls = 'H3_linear'
    else:
        cls = 'H0_no_verdict'
    det['verdict_class'] = cls
    return cls, det


clsD1, detD1 = curve_class(D1, rbar=1.0, tag='D1')
clsD2, detD2 = curve_class(D2, rbar=RBAR, tag='D2')

# amp 曲线分类（同一判据，作用在 amp 上）
amp_pts = [(r['alpha'] * RBAR, r['amp']) for r in D2 if r['amp'] is not None]
amp_cls, detAmp = 'H0_no_verdict', {}
if len(amp_pts) >= 3:
    xa = np.array([p[0] for p in amp_pts]); ya = np.array([p[1] for p in amp_pts])
    m = xa >= 0.10
    xf, yf = xa[m], ya[m]
    b1, b0 = np.polyfit(xf, yf, 1)
    ss = 1.0 - np.sum((yf - (b0 + b1 * xf)) ** 2) / max(np.sum((yf - yf.mean()) ** 2), 1e-12)
    pa = np.polyfit(np.log(xf), np.log(yf), 1) if np.all(yf > 0) else (None, None)
    ss_p = (1.0 - np.sum((np.log(yf) - (pa[1] + pa[0] * np.log(xf))) ** 2) /
            max(np.sum((np.log(yf) - np.log(yf).mean()) ** 2), 1e-12)) if pa[0] is not None else None
    sa = np.diff(ya) / np.diff(xa)
    ka = int(np.argmax(sa)); s_med_a = float(np.median(np.delete(sa, ka)))
    Ja = float(sa[ka] / s_med_a) if s_med_a > 0 else float('inf')
    amp_cls = 'H1_threshold' if Ja >= 5 and s_med_a > 0 else (
        'H2_powerlaw' if (ss_p is not None and ss_p >= 0.98 and pa[0] >= 2) else (
            'H3_linear' if (ss >= 0.97 and pa[0] is not None and pa[0] <= 1.5) else 'H0_no_verdict'))
    detAmp = dict(x=xa.tolist(), y=ya.tolist(), jump_ratio=Ja, lin_R2=float(ss),
                  gamma=(float(pa[0]) if pa[0] is not None else None), pow_R2=ss_p,
                  amp_ref=1.0 / RBAR, verdict_class=amp_cls)

# 组合裁决
combo = {
    ('H1_threshold', 'H1_threshold'): '阈值增益定位在 L6 内部（强支持 Phase 8 §7）',
    ('H1_threshold', 'H0_no_verdict'): '阈值在读出链；L6 自增益未见突变',
}.get((clsD1, amp_cls), None)
if combo is None:
    if clsD1 == 'H1_threshold':
        combo = 'D1 阈值 + amp 非阈值 => 阈值来自 L6 之后的读出链；§7 的"层内"定位须修正'
    elif clsD2 == 'H4_unreachable':
        combo = '上游末位 U 分量不可达 => 写入不由末位上游残差驱动（改判为上下文注意力驱动）'
    elif clsD1 in ('H2_powerlaw', 'H3_linear'):
        combo = '写入->行为为连续（%s）=> 支持"放大"、否证"突变"这一具体形式' % clsD1
    else:
        combo = '无单一判决（H0）：剂量曲线介于阈值与线性之间，须带噪声带重测'

floors = dict(D4=[r['dDonor'] for r in D4],
              D4_max=float(max(abs(r['dDonor']) for r in D4)),
              maxabs_all=float(maxabs),
              F1_ok=bool(max(abs(r['dDonor']) for r in D4) < 0.10 * maxabs),
              F2_ok=(not bad_base), F3_ok=bool(f3['L6out'] < 1e-2 and f3['L5out'] < 1e-2),
              F6_anchor=dict(donor=D1a['dDonor'], ref=ANCHOR_REF, drift=bool(ANCHOR_DRIFT)))
offmanifold = sorted(set([r['alpha'] for r in D2 if (r['pert_rel'] or 0) > PERT_LIM]))
# 描述性（不参与判决）：同剂量地板比 |D4(a)| / |D1(a)|
_d1m = {r['alpha']: r['dDonor'] for r in D1}
floor_matched = {}
for r in D4:
    den = _d1m.get(r['alpha'], 0.0)
    floor_matched[str(r['alpha'])] = (abs(r['dDonor']) / abs(den)) if abs(den) > 1e-9 else None
# D7 必要性读数（描述性，供组合解释）
_d7_0 = D7[[r['alpha'] for r in D7].index(0.0)] if 0.0 in [r['alpha'] for r in D7] else D7[0]
_d7_1 = D7[[r['alpha'] for r in D7].index(1.0)] if 1.0 in [r['alpha'] for r in D7] else D7[-1]
d7_identity_dev = abs(_d7_1['dD'])
KILL = _d7_0['kill_frac']

w('')
w('=== 判据 ===')
for tag, cls, det in [('D1 (写入->行为)', clsD1, detD1), ('D2 (上游->行为)', clsD2, detD2),
                      ('amp (L6 自增益)', amp_cls, detAmp)]:
    w('-- %s : %s' % (tag, cls))
    if not det:
        w('   （有效点不足，无法给出曲线细节）'); continue
    w('   x = %s' % ' '.join('%.3f' % v for v in det['x']))
    w('   y = %s' % ' '.join('%+.3f' % v for v in det['y']))
    if det.get('slopes'):
        w('   相邻斜率 = %s' % ' '.join('%+.2f' % v for v in det['slopes']))
    w('   jump_ratio J = %.2f ; x* = %s' % (det['jump_ratio'],
                                            ('%.4f' % det['x_star']) if det.get('x_star') else 'n/a'))
    if det.get('lin', {}).get('R2') is not None:
        w('   线性: slope=%+.3f R2=%.4f' % (det['lin']['slope'], det['lin']['R2']))
    if det.get('pow', {}).get('R2') is not None:
        w('   幂律: gamma=%.3f c=%.3f R2=%.4f' % (det['pow']['gamma'], det['pow']['c'], det['pow']['R2']))
w('')
w('full = dDonor_D1(a=1.0) = %+.3f  （Phase 8 diff6 臂 = +10.575，内建复现点）' % FULL)
w('地板: D4_max=%.4f ; maxabs_all=%.3f ; 比=%.4f ; F1_ok=%s' %
  (floors['D4_max'], maxabs, floors['D4_max'] / max(maxabs, 1e-9), floors['F1_ok']))
w('  同剂量地板比 |D4(a)|/|D1(a)| = %s  [描述性，不参与判决]' %
  '  '.join('%s:%.3f' % (k, v) for k, v in floor_matched.items() if v is not None))
w('D7 必要性: dD(alpha=0)=%+.3f (kill_frac=%s) ; dD(alpha=1)=%+.3f (恒等偏差 %.3e)' %
  (_d7_0['dD'], ('%.3f' % KILL) if KILL is not None else 'n/a', _d7_1['dD'], d7_identity_dev))
w('  读法: kill_frac ~ 1 => 上游 U 分量是写入的必要驱动；~ 0 => 上游 U 分量非必要（写由上下文驱动）')
w('离流形警告（pert_rel > %.2f 的 alpha）: %s' % (PERT_LIM, offmanifold if offmanifold else 'NONE'))
w('>>> 组合裁决: %s' % combo)
w('')

# ---------------- 9. 确认集 ----------------
conf_out = {}
if CONF:
    w('--- 确认集验带（n=%d，只跑 C1/C2）---' % len(CONF))
    CP = [p for p in PAIRS if p[0] in [x[0] for x in CONF]]
    C1 = run_dose(PRIMARY, grid_of('C1_conf_D1'), lambda rw, a: VEC[rw]['h6r'] + a * VEC[rw]['u6'], CP)
    C2 = run_dose(PRE, grid_of('C2_conf_D2'), lambda rw, a: VEC[rw]['h5r'] + a * VEC[rw]['u5'], CP)
    a1 = [r['alpha'] for r in C1]
    full_c = C1[a1.index(1.0)]['dDonor'] if 1.0 in a1 else float('nan')
    def curve_class_rel(dose, fullref, rbar, tag):
        xs = np.array([r['alpha'] * rbar for r in dose], float)
        ys = np.array([r['dDonor'] / fullref for r in dose], float)
        m = xs >= 0.01; xs2, ys2 = xs[m], ys[m]
        s = np.diff(ys2) / np.diff(xs2)
        k = int(np.argmax(s)); s_med = float(np.median(np.delete(s, k)))
        J = float(s[k] / s_med) if s_med > 0 else float('inf')
        mf = xs >= 0.10; xf, yf = xs[mf], ys[mf]
        lin = np.polyfit(xf, yf, 1) if len(xf) >= 3 else (None, None)
        R2l = (1 - np.sum((yf - (lin[1] + lin[0] * xf)) ** 2) / max(np.sum((yf - yf.mean()) ** 2), 1e-12)) \
            if lin[0] is not None else None
        pw = np.polyfit(np.log(xf), np.log(yf), 1) if (len(xf) >= 3 and np.all(yf > 0)) else (None, None)
        R2p = (1 - np.sum((np.log(yf) - (pw[1] + pw[0] * np.log(xf))) ** 2) /
               max(np.sum((np.log(yf) - np.log(yf).mean()) ** 2), 1e-12)) if pw[0] is not None else None
        h1 = bool(J >= 5 and s_med > 0 and float(ys2[k + 1]) >= 0.50 and float(np.max(ys2[:k + 1])) <= 0.15)
        if tag == 'D2' and float(np.max(np.abs(ys))) < 0.15:
            cls = 'H4_unreachable'
        elif h1: cls = 'H1_threshold'
        elif R2p is not None and R2p >= 0.98 and pw[0] >= 2: cls = 'H2_powerlaw'
        elif R2l is not None and R2l >= 0.97 and pw[0] is not None and pw[0] <= 1.5: cls = 'H3_linear'
        else: cls = 'H0_no_verdict'
        return cls, dict(J=J, R2_lin=R2l, gamma=(float(pw[0]) if pw[0] is not None else None),
                         R2_pow=R2p, y_max=float(np.max(np.abs(ys))))
    cD1, dC1 = curve_class_rel(C1, full_c, 1.0, 'D1')
    cD2, dC2 = curve_class_rel(C2, full_c, RBAR, 'D2')
    ratio = full_c / FULL if abs(FULL) > 1e-9 else float('nan')
    C3 = []
    for a in grid_of('C3_conf_D7'):
        acc, n = 0.0, 0
        for (rw, rs, dw, ds, sw) in CP:
            if dw not in DB:
                continue
            V = VEC[rw]; sid_d = ids_of(dw)[0]
            vec = CAP[dw][0][H5].astype(np.float32) - (1.0 - a) * V['u5']
            lg = fwd_patch(TMPL % dw, PRE, torch.tensor(vec, device='cuda'))
            acc += score_of(lg, ds, sid_d) - DB[dw]['sD1']; n += 1
        C3.append(dict(alpha=float(a), dD=acc / max(n, 1), n=n))
    w('  C3 (D7 验带, 供体句): %s' %
      '  '.join('a=%.2f dD=%+.3f' % (r['alpha'], r['dD']) for r in C3))
    conf_out = dict(full_conf=full_c, full_ratio=float(ratio), C3=C3,
                    C1=dict(cls=cD1, **dC1), C2=dict(cls=cD2, **dC2),
                    same_verdict_D1=bool(cD1 == clsD1), same_verdict_D2=bool(cD2 == clsD2))
    w('  确认集 full=%.3f（发现集 %.3f，比 %.3f）' % (full_c, FULL, ratio))
    w('  C1 %s (J=%.2f R2lin=%s gamma=%s) ; 与发现集同判=%s' %
      (cD1, dC1['J'], ('%.4f' % dC1['R2_lin']) if dC1['R2_lin'] is not None else 'n/a',
       ('%.2f' % dC1['gamma']) if dC1['gamma'] is not None else 'n/a', conf_out['same_verdict_D1']))
    w('  C2 %s (J=%.2f y_max=%.3f) ; 与发现集同判=%s' %
      (cD2, dC2['J'], dC2['y_max'], conf_out['same_verdict_D2']))

el = time.time() - t0
w('')
w('total %.1fs' % el)

# ---------------- 10. 落盘 ----------------
res = dict(
    phase=9, name='N2h1-alpha-2/threshold_gain_dose_response', model=MODEL, smoke=SMOKE,
    elapsed_s=round(el, 1),
    layers=dict(L=L, primary=PRIMARY, pre=PRE, n_heads=NH, head_dim=HD),
    panel=dict(discovery=len(DISC), confirmation=len(CONF)),
    dose_coord=dict(mean_n6=float(_n6), mean_n5=float(_n5), rbar=RBAR, rbar_phase8=RBAR_REF,
                    rbar_drift=bool(RBAR_DRIFT), r_per_pair=dict(mean=float(RPER.mean()),
                                                                 sd=float(RPER.std()),
                                                                 min=float(RPER.min()),
                                                                 max=float(RPER.max())),
                    mean_n5b=float(np.mean([VEC[rw]['n5b'] for rw in VEC])),
                    mean_nd6=float(np.mean([VEC[rw]['nd6'] for rw in VEC])),
                    mean_nh5=float(np.mean([VEC[rw]['nh5'] for rw in VEC])),
                    amp_ref=float(1.0 / RBAR)),
    subspace=dict(u5_u6_principal_cos=[float(x) for x in sv_c], overlap=overlap,
                  sing_U6=[float(x) for x in SV6], n_classes=nA6),
    base=dict(ok=(not bad_base), recip_mean=float(np.mean([b['sr0'] for b in BASE.values()])),
              donor_mean=float(np.mean([b['sd0'] for b in BASE.values()])),
              donor_rank1_frac=base_r1),
    full=FULL,
    D1=D1, D1a=D1a, D2=D2, D2b=D2b, D3=D3, D4=D4, D6=D6, D7=D7,
    D7_detail=dict(ref_drop=REF_D7, dD_alpha0=_d7_0['dD'], kill_frac=KILL,
                   identity_dev=d7_identity_dev,
                   donor_rank1=float(np.mean([1.0 if DB[p[2]]['rD1'] == 1 else 0.0 for p in _D7P])),
                   sD1_mean=float(np.mean([DB[p[2]]['sD1'] for p in _D7P]))),
    floor_matched=floor_matched,
    curves=dict(D1=detD1, D2=detD2, amp=detAmp),
    verdict=dict(D1=clsD1, D2=clsD2, amp=amp_cls, combo=combo),
    floors=dict(D4_max=floors['D4_max'], maxabs_all=floors['maxabs_all'], F1_ok=floors['F1_ok'],
                F2_ok=floors['F2_ok'], F3_ok=floors['F3_ok'],
                F3_dev=dict(f3), F6_anchor=floors['F6_anchor']),
    off_manifold_alphas=offmanifold,
    confirmation=conf_out,
    drift_flags=dict(rbar_drift=bool(RBAR_DRIFT), anchor_drift=bool(ANCHOR_DRIFT),
                     off_manifold=offmanifold),
    seal_sha8=sha(SEAL)[:8], exec_sha8=sha(EXEC)[:8],
    amend1='N2h1a2-amend1 (D7 必要性对偶臂 + detAmp 守卫 + 同剂量地板比；阈值/面板未动)',
)
io.open(RESULT, 'w', encoding='utf-8').write(json.dumps(res, ensure_ascii=False, indent=1))
io.open(REPORT, 'w', encoding='utf-8').write('\n'.join(lines))
print('DONE ->', REPORT, '|', RESULT)
