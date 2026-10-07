# -*- coding: utf-8 -*-
"""
Phase 14 / N2h1-alpha-7 : 逐层累积全位点代换（cumulative prefix substitution）
=============================================================================
预注册：tests/deepseek_temp/Phase14/N2h1a7_design_seal.json（观测前冻结，sha8 074fc963）
执行冻结：tests/deepseek_temp/Phase14/execution_phase14.json（sha8 2a5699c4）

要回答的问题（Phase 13 §8 死线，最高优先）：
  Phase 8-13 的全部探针族都只替换【末位】残差。本 Phase 把 mask 扩到 positions{0,1}
  （整段前缀，T=2 由预 seal 探针确证），得到第三条独立口径的 xhalf_p / J_p；
  并按铁律 (t) 在 J 与 xhalf 两个坐标上同时报告 top3_share 与 argmax 窗口，
  与 Phase 13 已发表的两坐标 argmax（xhalf=14 / J_swap=1）做跨族迁移检验。

用法：python run_phase14.py smoke | python run_phase14.py formal
"""
import os, sys, io, json, time, hashlib
import numpy as np
import torch

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P14T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase14')
SEAL = os.path.join(P14T, 'N2h1a7_design_seal.json')
AMEND = os.path.join(P14T, 'N2h1a7_design_seal_amend1.json')
AMEND2 = os.path.join(P14T, 'N2h1a7_design_seal_amend2.json')
EXEC = os.path.join(P14T, 'execution_phase14.json')
REPORT = os.path.join(P14T, 'n2h1a7_report_qwen3-4b.txt')
RESULT = os.path.join(P14T, 'result_phase14.json')
SMOKE = os.environ.get('SMOKE', '0') == '1'
if SMOKE:
    _S = os.path.join(P14T, 'smoke')
    os.makedirs(_S, exist_ok=True)
    REPORT = os.path.join(_S, 'n2h1a7_report_qwen3-4b.txt')
    RESULT = os.path.join(_S, 'result_phase14.json')

lines = []
def w(s=''):
    lines.append(str(s)); print(s); sys.stdout.flush()


def sha(p):
    return hashlib.sha256(io.open(p, 'rb').read()).hexdigest()


E = json.load(io.open(EXEC, encoding='utf-8'))
S = json.load(io.open(SEAL, encoding='utf-8'))
AM = json.load(io.open(AMEND, encoding='utf-8'))
AM2 = json.load(io.open(AMEND2, encoding='utf-8'))

_REQ = ['seal_sha256', 'seal_sha8', 'amend1', 'amend2', 'model', 'template', 'seed', 'classes', 'sup_id', 'panel',
        'sites', 'masks', 'alpha_grids', 'stats', 'bootstrap', 'arms', 'expected_cfg',
        'n_heads', 'head_dim', 'n_kv_heads', 'o_proj_in_features', 'inherits', 'decision', 'floors',
        'result_keys', 'expected_fwd_total', 'off_manifold_pert_rel']
_miss = [k for k in _REQ if k not in E]
assert not _miss, 'execution 缺字段: %s' % _miss
assert sha(SEAL) == E['seal_sha256'], 'F 前置：seal 已漂移'
assert sha(AMEND) == E['amend1']['sha256'], 'F 前置：amend1 已漂移'
assert AM['amend_of_seal_sha256'] == E['seal_sha256'], 'amend1 指向的 seal 与本 exec 不一致'
assert sha(AMEND2) == E['amend2']['sha256'], 'F 前置：amend2 已漂移'
assert AM2['amend_of_seal_sha256'] == E['seal_sha256'], 'amend2 指向的 seal 与本 exec 不一致'
assert AM2['amend1_sha8'] == sha(AMEND)[:8], 'amend2 记录的 amend1 哈希不一致'

MODEL = E['model']
MDIR = os.path.join(ROOT, E['model_dir'])
TMPL = E['template']
SUP_ID = {k: int(v) for k, v in E['sup_id'].items()}
SUPS = list(E['classes'])
SITES = [int(x) for x in E['sites']['profile']]
CONF_SITES = [int(x) for x in E['sites']['conf']]
FLOOR_SITES = [int(x) for x in E['sites']['floor']]
F3_SITES = E['sites']['f3']
R_SITE = E['sites']['readout']
AL_LEG = list(E['alpha_grids']['legacy'])
AL_DEN = list(E['alpha_grids']['dense'])
AL_CONF = list(E['alpha_grids']['conf'])
XHF = float(E['stats']['xhalf_frac'])
JFL = float(E['stats']['jdose_floor'])
W = int(E['stats']['window_W'])
N_WIN = int(E['stats']['n_windows'])
BS = int(E['bootstrap']['BS'])
BP = int(E['bootstrap']['BP'])
SEED = int(E['bootstrap']['seed'])
NH, HD = E['n_heads'], E['head_dim']
OIN_EXP = E['o_proj_in_features']
PERT_LIM = float(E['off_manifold_pert_rel'])
N6_TOL = float(E['n6_drift_tol'])
ANCHOR_TOL = float(E['anchor_drift_tol'])
INH = E['inherits']
PRIMARY = 6
H6 = PRIMARY + 1

DISC = [tuple(x) for x in E['panel']['discovery']]
CONF = [tuple(x) for x in E['panel']['confirmation']]
INST_ALL = [tuple(x) for x in E['panel']['instances_all']]
PAIRS_ALL = [tuple(p) for p in E['panel']['pairs_all']]

w('=== Phase 14 / N2h1-alpha-7 : cumulative prefix (all-position) substitution ===')
w('smoke=%s ; time %s' % (SMOKE, time.strftime('%Y-%m-%d %H:%M:%S')))
w('seal sha8 %s ; amend1 sha8 %s ; exec sha8 %s ; exec.seal_sha8 %s' %
  (sha(SEAL)[:8], sha(AMEND)[:8], sha(EXEC)[:8], E['seal_sha8']))
w('amend1: %s (%s) ; amend_of_seal_sha8 %s' %
  (AM['kind'], AM['trigger'], AM['amend_of_seal_sha8']))
w('配置（amend1 直读地面真值）: heads=%d kv_heads=%d head_dim=%d o_proj_in=%d GQA=%s' %
  (E['n_heads'], E['n_kv_heads'], E['head_dim'], E['o_proj_in_features'],
   AM['discovered_ground_truth'].get('is_gqa')))
w('exec inherits phase12 result sha8 %s ; phase13 result sha8 %s' %
  (INH['phase12_result_sha8'], INH['phase13_result_sha8']))
sys.stdout.flush()

# ---------------------------------------------------------------- 继承锚
R12P = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase12', 'result_phase12.json')
R13P = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase13', 'result_phase13.json')
assert sha(R12P) == INH['phase12_result_sha256'], 'F 前置：result_phase12.json 已漂移'
assert sha(R13P) == INH['phase13_result_sha256'], 'F 前置：result_phase13.json 已漂移'
R12 = json.load(io.open(R12P, encoding='utf-8'))
R13 = json.load(io.open(R13P, encoding='utf-8'))

FULL_SWAP_INH = float(INH['FULL_SWAP'])
N6_REF = float(INH['mean_n6_ref_phase9'])
SING_U6_INH = [float(x) for x in INH['sing_U6']]
REC_12 = {int(k): float(v) for k, v in INH['recover_12_by_site'].items()}
XH_12 = {int(k): float(v) for k, v in INH['XH_12_by_site'].items()}
J_12 = {int(k): float(v) for k, v in INH['J_swap_12_by_site'].items()}
XH_RANGE_12 = float(INH['XH_RANGE_12'])
Q_ELL_12 = {int(k): float(v) for k, v in INH['Q_ELL_12'].items()}
MODE_X_13 = int(INH['MODE_X_13'])
MODE_J_13 = int(INH['MODE_J_13'])

# ---------------------------------------------------------------- F24 FULL_SWAP 重建
FS_KEY = R12['E2']['6'][0]['order']
FS_VEC = np.array([R12['FULL_SWAP_pairs'][x] for x in FS_KEY], float)
FULL_SWAP = float(np.mean(FS_VEC))
F24_ok = (FULL_SWAP == FULL_SWAP_INH)
w('')
w('--- F24 FULL_SWAP 重建（零前向）---')
w('  rebuilt = %.15f ; inherited = %.15f ; bit-equal = %s' % (FULL_SWAP, FULL_SWAP_INH, F24_ok))

# ---------------------------------------------------------------- 模型
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
assert NORM is not None

drift = []
if L != E['expected_cfg']['num_hidden_layers']: drift.append('layers')
if HID != E['expected_cfg']['hidden_size']: drift.append('hidden')
if CFG.num_attention_heads != NH: drift.append('n_heads')
if int(getattr(CFG, 'head_dim', -1)) != HD: drift.append('head_dim(%s)' % getattr(CFG, 'head_dim', None))
if int(getattr(CFG, 'num_key_value_heads', NH)) != int(E['n_kv_heads']): drift.append('n_kv_heads')
if bool(CFG.tie_word_embeddings) != bool(E['expected_cfg']['tie_word_embeddings']): drift.append('tie')
ATTN0 = None
for nm in ['o_proj', 'dense', 'out_proj']:
    if hasattr(layers[PRIMARY].self_attn, nm):
        ATTN0 = getattr(layers[PRIMARY].self_attn, nm); break
if ATTN0 is None or ATTN0.in_features != OIN_EXP: drift.append('o_proj_in')
w('')
w('model=%s L=%d hid=%d heads=%d kv_heads=%d head_dim=%d tie=%s ; load %.1fs' %
  (MODEL, L, HID, NH, int(getattr(CFG, 'num_key_value_heads', NH)), HD, CFG.tie_word_embeddings,
   time.time() - t0))
w('o_proj(%s).in_features = %s (expect %d = %d*%d)' %
  (type(ATTN0).__name__ if ATTN0 is not None else 'None',
   ATTN0.in_features if ATTN0 is not None else None, OIN_EXP, NH, HD))
w('drift: %s' % (drift if drift else 'NONE'))
assert not drift, 'DRIFT 非空，按预注册停止'
assert max(SITES) < L - 1, 'profile sites 须剔末层'


def ids_of(s):
    return tok.encode(s, add_special_tokens=False)


# ---------------------------------------------------------------- F27 T == 2
TIDS = {wd: ids_of(TMPL % wd) for wd, _ in INST_ALL}
T_ALL = sorted(set(len(v) for v in TIDS.values()))
w('')
w('--- F27 template 布局 ---')
w('  distinct T across %d instances = %s ; sample %s -> %s' %
  (len(TIDS), T_ALL, INST_ALL[0][0], [tok.decode([i]) for i in TIDS[INST_ALL[0][0]]]))
F27_ok = (T_ALL == [2])


# ---------------------------------------------------------------- capture
@torch.no_grad()
def capture(text):
    ii = torch.tensor([ids_of(text)], device='cuda')
    rec = {}

    def hk(mod, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        rec['hR_all'] = t[0].float().detach().cpu().numpy()   # [T, HID]
        return out

    h = NORM.register_forward_hook(hk)
    try:
        o = model(input_ids=ii, output_hidden_states=True)
    finally:
        h.remove()
    HH = np.stack([x[0].float().detach().cpu().numpy() for x in o.hidden_states], 0)  # [L+1,T,HID]
    return HH, rec['hR_all'], o.logits[0, -1].float().detach().cpu().numpy()


t_cap = time.time()
CAP = {}
for wd, sup in INST_ALL:
    CAP[wd] = capture(TMPL % wd)
w('')
w('capture %d instances in %.2fs ; hidden shape %s ; hR_all shape %s' %
  (len(CAP), time.time() - t_cap, CAP[INST_ALL[0][0]][0].shape, CAP[INST_ALL[0][0]][1].shape))


def proj(vec, Ub):
    return (vec @ Ub.T) @ Ub


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
F26_dev = float(np.max(np.abs(SV6 - np.array(SING_U6_INH)) / np.maximum(np.abs(SING_U6_INH), 1e-12)))
F26_ok = F26_dev <= 1e-6
w('')
w('--- F26 U6 重建 ---')
w('  sing = %s' % ' '.join('%.6f' % x for x in SV6))
w('  inherited = %s' % ' '.join('%.6f' % x for x in SING_U6_INH))
w('  max relative dev = %.3e ; ok = %s' % (F26_dev, F26_ok))

PAIRS = [p for p in PAIRS_ALL if p[0] in CAP and p[2] in CAP]
PAIRS_D = [p for p in PAIRS if p[0] in [x[0] for x in DISC]]
CONF_D = [p for p in PAIRS if p[0] in [x[0] for x in CONF]]


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
F31_ok = (len(bad_base) == 0)
w('')
w('--- F31 base ---')
w('  n=%d ; 受体类分数 mean %+.4f ; 供体类已 rank1 比例 %.3f ; base<=0 : %s' %
  (len(BASE), np.mean([b['sr0'] for b in BASE.values()]),
   float(np.mean([1.0 if BASE[rw]['rd0'] == 1 else 0.0 for rw in BASE])), bad_base if bad_base else 'NONE'))
assert F31_ok, 'F31 失败'

# ---------------------------------------------------------------- VEC（全位置）
H_SITES = sorted(set(SITES + [PRIMARY]))
VEC = {}
for (rw, rs, dw, ds, sw) in PAIRS:
    d6 = CAP[dw][0][H6][-1].astype(np.float32) - CAP[rw][0][H6][-1].astype(np.float32)
    u6 = proj(d6, AO)
    n6 = float(np.linalg.norm(u6))
    d = dict(u6=u6, n6=n6,
             hR_all=CAP[rw][1].astype(np.float32), hR_all_donor=CAP[dw][1].astype(np.float32),
             h_ell={}, h_ell0={}, hd_ell={}, hd_ell0={},
             d_ell={}, d_ell0={}, nh_ell={}, nh_ell0={}, nhR={})
    for s in H_SITES:
        hr, hd = CAP[rw][0][s + 1], CAP[dw][0][s + 1]           # [T, HID]
        d['h_ell'][s] = hr[-1].astype(np.float32)
        d['h_ell0'][s] = hr[0].astype(np.float32)
        d['hd_ell'][s] = hd[-1].astype(np.float32)
        d['hd_ell0'][s] = hd[0].astype(np.float32)
        d['d_ell'][s] = (hd[-1] - hr[-1]).astype(np.float32)
        d['d_ell0'][s] = (hd[0] - hr[0]).astype(np.float32)
        d['nh_ell'][s] = float(np.linalg.norm(hr[-1]))
        d['nh_ell0'][s] = float(np.linalg.norm(hr[0]))
    d['nhR'] = float(np.linalg.norm(CAP[rw][1][-1]))
    VEC[rw] = d

# F25 n6 锚
N6_MEAN = float(np.mean([VEC[rw]['n6'] for rw in VEC]))
F25_dev = abs(N6_MEAN - N6_REF)
F25_ok = F25_dev <= N6_TOL
# F35 q_ell 锚
qell_dev = 0.0
for s in SITES:
    q = float(np.mean([np.linalg.norm(VEC[rw]['d_ell'][s]) / max(VEC[rw]['nh_ell'][s], 1e-9)
                       for rw in VEC]))
    qell_dev = max(qell_dev, abs(q - Q_ELL_12[s]))
F35_ok = qell_dev <= 1e-6
w('')
w('--- F25/F35 剂量锚 ---')
w('  mean||P_U6(diff6)|| = %.12f ; ref %.12f ; |d| = %.3e (tol %.1e) ; ok=%s' %
  (N6_MEAN, N6_REF, F25_dev, N6_TOL, F25_ok))
w('  max_ell |q_ell - Phase12| = %.3e ; ok=%s' % (qell_dev, F35_ok))


def srcs_of(V, site):
    if site == R_SITE:
        return {0: V['hR_all'][0], 1: V['hR_all'][-1]}
    return {0: V['h_ell0'][site], 1: V['h_ell'][site]}


def diffs_of(V, site):
    if site == R_SITE:
        return {0: (V['hR_all_donor'][0] - V['hR_all'][0]).astype(np.float32),
                1: (V['hR_all_donor'][-1] - V['hR_all'][-1]).astype(np.float32)}
    return {0: V['d_ell0'][site], 1: V['d_ell'][site]}


def nh_of(V, site):
    if site == R_SITE:
        return max(V['nhR'], 1e-9)
    return max(V['nh_ell'][site], 1e-9)


# ---------------------------------------------------------------- patch
@torch.no_grad()
def fwd_patch(text, site, vecs_by_pos):
    ii = torch.tensor([ids_of(text)], device='cuda')
    mod = NORM if site == R_SITE else layers[site]

    def hook(m, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        t = t.clone()
        Tt = t.shape[1]
        for p, v in vecs_by_pos.items():
            pp = Tt - 1 if p < 0 else p
            t[0, pp, :] = torch.tensor(v, device=t.device, dtype=t.dtype)
        return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t

    h = mod.register_forward_hook(hook)
    try:
        o = model(input_ids=ii)
    finally:
        h.remove()
    return o.logits[0, -1].float().detach().cpu().numpy()


def _acc(lg, B, rw, rs, dw, ds):
    sid_r, sid_d = ids_of(rw)[0], ids_of(dw)[0]
    return (score_of(lg, ds, sid_d) - B['sd0'],
            score_of(lg, rs, sid_r) - B['sr0'],
            1 if rank_of(lg, ds, sid_d) == 1 else 0)


# ---------------------------------------------------------------- F28 alpha=0 noop
w('')
w('--- F28 alpha=0 全位点 patch == capture ---')
F28_dev = 0.0; F28_detail = {}
for site in F3_SITES:
    devs = []
    for (rw, rs, dw, ds, sw) in PAIRS_D[:3]:
        V = VEC[rw]
        _v = srcs_of(V, site)
        lg = fwd_patch(TMPL % rw, site, {p: _v[p] for p in (0, 1)})
        devs.append(abs(score_of(lg, rs, ids_of(rw)[0]) - BASE[rw]['sr0']))
    F28_detail[str(site)] = float(max(devs))
    F28_dev = max(F28_dev, float(max(devs)))
F28_ok = F28_dev < 1e-2
w('  max|dScore| : %s ; ok=%s' %
  ('  '.join('%s=%.3e' % (k, v) for k, v in F28_detail.items()), F28_ok))
sys.stdout.flush()


# ---------------------------------------------------------------- 剂量函数
def dose_mask(site, alphas, pairs, mask, per_pair=True):
    out = []
    for a in alphas:
        dd = dr = 0.0; r1 = 0; n = 0; per = []; order = []; perts = []
        for (rw, rs, dw, ds, sw) in pairs:
            V, B = VEC[rw], BASE[rw]
            src = srcs_of(V, site); dif = diffs_of(V, site)
            vecs = {p: (src[p] + a * dif[p]).astype(np.float32) for p in mask}
            lg = fwd_patch(TMPL % rw, site, vecs)
            x1, x2, x3 = _acc(lg, B, rw, rs, dw, ds)
            dd += x1; dr += x2; r1 += x3; n += 1
            per.append(float(x1)); order.append(rw)
            nd = float(np.mean([np.linalg.norm(dif[p]) for p in mask])) if mask else 0.0
            perts.append(a * nd / nh_of(V, site))
        row = dict(alpha=float(a), dDonor=dd / max(n, 1), dRecip=dr / max(n, 1),
                   rank1=r1 / max(n, 1), n=n, pert_rel=float(np.mean(perts)) if perts else 0.0)
        if per_pair:
            row['per_pair'] = per; row['order'] = order
        out.append(row)
    return out


def curve_from_rows(rows, key='dDonor'):
    return ([float(r['alpha']) for r in rows], [float(r[key]) / FULL_SWAP for r in rows])


@torch.no_grad()
def fwd_patch_multi(text, sites_list, vecs_by_layer):
    """在【多个】层上同时挂钩（A8 逐层累积层代换用）。
    sites_list: [j,...]（层的下标）；vecs_by_layer: {j: {pos: np.array(HID)}}。"""
    ii = torch.tensor([ids_of(text)], device='cuda')
    handles = []

    def mk(js):
        def hook(m, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            t = t.clone()
            Tt = t.shape[1]
            for p, v in vecs_by_layer[js].items():
                pp = Tt - 1 if p < 0 else p
                t[0, pp, :] = torch.tensor(v, device=t.device, dtype=t.dtype)
            return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t
        return hook

    for j in sites_list:
        handles.append(layers[j].register_forward_hook(mk(j)))
    try:
        o = model(input_ids=ii)
    finally:
        for h in handles:
            h.remove()
    return o.logits[0, -1].float().detach().cpu().numpy()


def dose_cumlayer(ell, sup, alphas, pairs, per_pair=True):
    """A8：对支撑序对应的层集合 sup 的【末位】同时写入 h_j + alpha*d_j。"""
    out = []
    for a in alphas:
        dd = dr = 0.0; r1 = 0; n = 0; per = []; order = []; perts = []
        for (rw, rs, dw, ds, sw) in pairs:
            V, B = VEC[rw], BASE[rw]
            vbl = {j: {-1: (V['h_ell'][j] + a * V['d_ell'][j]).astype(np.float32)} for j in sup}
            lg = fwd_patch_multi(TMPL % rw, sup, vbl)
            x1, x2, x3 = _acc(lg, B, rw, rs, dw, ds)
            dd += x1; dr += x2; r1 += x3; n += 1
            per.append(float(x1)); order.append(rw)
            perts.append(a * float(np.mean([np.linalg.norm(V['d_ell'][j]) for j in sup]))
                         / nh_of(V, sup[-1]))
        row = dict(alpha=float(a), dDonor=dd / max(n, 1), dRecip=dr / max(n, 1),
                   rank1=r1 / max(n, 1), n=n, pert_rel=float(np.mean(perts)) if perts else 0.0,
                   support=list(sup))
        if per_pair:
            row['per_pair'] = per; row['order'] = order
        out.append(row)
    return out


# ---------------------------------------------------------------- 统计量（逐字复制 Phase 12/13）
def _rank(a):
    a = np.asarray(a, float)
    o = np.argsort(a)
    r = np.empty(len(a), float)
    r[o] = np.arange(len(a), dtype=float)
    return r


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2:
        return None
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 1e-12 else 0.0


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


def J_only(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    m = xs >= JFL
    xs2, ys2 = xs[m], ys[m]
    if len(xs2) < 3:
        return np.nan
    s = np.diff(ys2) / np.diff(xs2)
    k_i = int(np.argmax(s))
    rest = np.delete(s, k_i)
    s_med = float(np.median(rest)) if len(rest) > 1 else 0.0
    return float(s[k_i] / s_med) if s_med > 1e-12 else float('inf')


def J_iqr(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    m = xs >= JFL
    xs2, ys2 = xs[m], ys[m]
    if len(xs2) < 3:
        return np.nan
    s = np.diff(ys2) / np.diff(xs2)
    k_i = int(np.argmax(s))
    rest = np.delete(s, k_i)
    if len(rest) < 3:
        return np.nan
    q1, q3 = np.percentile(rest, [25, 75])
    den = float(q3 - q1)
    return float(s[k_i] / den) if den > 1e-12 else float('inf')


def _ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return dict(lo=None, hi=None, med=None, n_ok=int(len(v)))
    lo, hi = np.percentile(v, [2.5, 97.5])
    return dict(lo=float(lo), hi=float(hi), med=float(np.median(v)), n_ok=int(len(v)))


def conc_hat(Fhat):
    F = np.asarray(Fhat, float)
    jm = np.diff(F)
    if not np.isfinite(F).all():
        return None, None, [float(v) for v in jm]
    rng = float(F.max() - F.min())
    if rng <= 1e-12 or len(jm) < W:
        return None, None, jm.tolist()
    wins = [abs(float(np.sum(jm[j:j + W]))) for j in range(len(jm) - W + 1)]
    k = int(np.argmax(wins))
    return float(wins[k] / rng), k, jm.tolist()


def logistic_fit(xs, ys, CL):
    from math import exp
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    ymax = float(np.max(ys))
    if not np.isfinite(ymax) or ymax <= 1e-9:
        return None
    yn = ys / ymax
    pen = 1e-6
    best = None
    kk = np.arange(CL['logistic_k_min'], CL['logistic_k_max'] + 1e-9, CL['logistic_k_step'])
    x0s = np.arange(0.0, 1.0 + 1e-9, CL['logistic_x0_step'])
    for k in kk:
        for x0 in x0s:
            p = 1.0 / (1.0 + np.exp(-k * (xs - x0)))
            # 加截距的自由缩放（最小二乘闭式）
            A = np.stack([p], 1)
            num = float((p * yn).sum()); den = float((p * p).sum()) + pen
            if den <= 0:
                continue
            a = num / den
            r = yn - a * p
            ss = float((r ** 2).sum())
            if best is None or ss < best[0]:
                best = (ss, float(k), float(x0), float(a))
    # R2 相对 yn
    sst = float(((yn - yn.mean()) ** 2).sum())
    r2 = 1.0 - best[0] / sst if sst > 1e-12 else 0.0
    return dict(k_log=best[1], x_star=best[2], amp=best[3], R2_log=float(r2))


def lin_r2(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    A = np.stack([xs, np.ones_like(xs)], 1)
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    pred = A @ coef
    ss = float(((ys - pred) ** 2).sum())
    sst = float(((ys - ys.mean()) ** 2).sum())
    return 1.0 - ss / sst if sst > 1e-12 else 0.0


# ================================================================ A1
if SMOKE:
    _A1_SITES = SITES[:3]
    _A1_AL = [0.0, 0.1, 0.2, 0.5, 0.8, 1.0]
    _A1_PAIRS = PAIRS_D[:6]
    CONF_D = []
    FLOOR_SITES = []
else:
    _A1_SITES = SITES
    _A1_AL = AL_DEN
    _A1_PAIRS = PAIRS_D

w('')
w('--- A1 全位点代换 层扫描（mask={0,1}，%d 位点 x %d alpha x %d 对）---' % (len(_A1_SITES), len(_A1_AL), len(_A1_PAIRS)))
t_a1 = time.time()
A1 = {}
for s in _A1_SITES:
    A1[str(s)] = dose_mask(s, _A1_AL, _A1_PAIRS, (0, 1), per_pair=True)
w('  A1 done %.1fs' % (time.time() - t_a1))
sys.stdout.flush()

A1_CURVES, A1_XH, A1_J, A1_Ji, A1_LOG, A1_PER = {}, {}, {}, {}, {}, {}
for s in _A1_SITES:
    xs, ys = curve_from_rows(A1[str(s)])
    A1_CURVES[str(s)] = dict(x=xs, y=ys,
                             pert_rel=[float(r['pert_rel']) for r in A1[str(s)]],
                             rank1=[float(r['rank1']) for r in A1[str(s)]])
    A1_PER[str(s)] = [list(map(float, r['per_pair'])) for r in A1[str(s)]]
    A1_XH[str(s)] = cross_alpha(xs, ys, XHF)
    A1_J[str(s)] = J_only(xs, ys)
    A1_Ji[str(s)] = J_iqr(xs, ys)
    A1_LOG[str(s)] = logistic_fit(xs, ys, S['curve_classifier'])

w('')
w('%-5s %-11s %-11s %-11s %-11s %-9s %-8s %-8s' %
  ('ell', 'xhalf_p', 'J_p', 'J_iqr_p', 'recover_p', 'R2_log', 'k_log', 'x*_log'))
for s in _A1_SITES:
    lg = A1_LOG[str(s)] or {}
    w('L%-4d %-11.6f %-11.4f %-11.4f %-11.6f %-9.4f %-8.1f %-8.4f' % (
        s, A1_XH[str(s)] if A1_XH[str(s)] is not None else float('nan'),
        A1_J[str(s)], A1_Ji[str(s)], A1_CURVES[str(s)]['y'][-1],
        lg.get('R2_log', float('nan')), lg.get('k_log', float('nan')), lg.get('x_star', float('nan'))))
w('  参照 Phase 12 单点族 : xhalf %s' %
  ' '.join('L%d:%.4f' % (s, XH_12[s]) for s in _A1_SITES))
w('  参照 Phase 12 单点族 : J     %s' %
  '  '.join('L%d:%.2f' % (s, J_12[s]) for s in _A1_SITES))
sys.stdout.flush()

# ================================================================ A3b 位置端点
w('')
w('--- A3b 位置端点（alpha=1），mask={0} 与 {1} 对 18 位点 ---')
t_a3b = time.time()
A3B = {}
_y0 = {}; _y1 = {}; _y01 = {}
for s in (_A1_SITES if SMOKE else SITES):
    for tag, mask in (('m0', (0,)), ('m1', (1,))):
        rows = dose_mask(s, [1.0], _A1_PAIRS, mask, per_pair=False)
        A3B.setdefault(str(s), {})[tag] = float(rows[0]['dDonor']) / FULL_SWAP
    _y0[s] = A3B[str(s)]['m0']
    _y1[s] = A3B[str(s)]['m1']
    _y01[s] = A1_CURVES[str(s)]['y'][-1] if str(s) in A1_CURVES else float('nan')
w('  A3b done %.1fs' % (time.time() - t_a3b))

F29_dev = 0.0; F30_dev = 0.0
F29_rows = []
for s in _y1:
    F29_rows.append((s, _y1[s], REC_12[s], abs(_y1[s] - REC_12[s])))
    F29_dev = max(F29_dev, abs(_y1[s] - REC_12[s]))
    F30_dev = max(F30_dev, abs(_y01[s] - 1.0))
w('%-5s %-13s %-13s %-13s %-11s %-11s' % ('ell', 'y0(={0})', 'y1(={1})', 'y01(={0,1})', 'Phase12 rec.', '|d| F29'))
for s in sorted(_y0):
    w('L%-4d %-13.6f %-13.6f %-13.6f %-11.6f %-11.3e' %
      (s, _y0[s], _y1[s], _y01[s], REC_12[s], abs(_y1[s] - REC_12[s])))
if _y0:
    rat = np.array([_y0[s] / _y1[s] for s in sorted(_y0) if _y1[s] > 1e-9])
    Sdd = np.array([_y0[s] + _y1[s] - _y01[s] for s in sorted(_y0)])
    w('  median y0/y1 = %.4f ; n(y0>y1) = %d/%d ; n(y0>0) = %d/%d' %
      (float(np.median(rat)), int(np.sum(np.array([_y0[s] for s in sorted(_y0)]) >
                                         np.array([_y1[s] for s in sorted(_y0)]))),
       len(_y0), int(np.sum(np.array([_y0[s] for s in sorted(_y0)]) > 0)), len(_y0)))
    w('  S = y0+y1-y01 : mean %+.4f ; n(S>0) = %d/%d ; min %+.4f ; max %+.4f' %
      (float(Sdd.mean()), int(np.sum(Sdd > 0)), len(Sdd), float(Sdd.min()), float(Sdd.max())))
else:
    rat = np.array([]); Sdd = np.array([])
FULL_PANEL = (len(_A1_PAIRS) == 24)
# F30(a) 与子集无关的逐对恒等式：alpha=1 时 per_pair == Phase12 FULL_SWAP_pairs[受体词]
F30a_dev = 0.0
_n_f30a = 0
for s in _A1_SITES:
    for row in A1[str(s)]:
        if abs(row['alpha'] - 1.0) > 1e-12:
            continue
        for k_, rw_ in enumerate(row['order']):
            ref = float(R12['FULL_SWAP_pairs'][rw_])
            F30a_dev = max(F30a_dev, abs(row['per_pair'][k_] - ref) / max(abs(ref), 1e-9))
            _n_f30a += 1
F30a_ok = bool(F30a_dev <= 1e-4)
# F30(b) 面板级常数性（仅在臂的 pair 集 == 24 个 discovery 对时断言）
_y01_sorted = np.array([_y01[s] for s in sorted(_y01)], float)
F30b_range = float(_y01_sorted.max() - _y01_sorted.min()) if len(_y01_sorted) else None
F30b_dev = (abs(float(_y01_sorted[0]) - 1.0) if len(_y01_sorted) else None)
if FULL_PANEL and F30b_range is not None:
    F30b_ok = bool(F30b_range <= 1e-12 and F30b_dev <= ANCHOR_TOL)
else:
    F30b_ok = None
F29_ok = bool(F29_dev <= ANCHOR_TOL) if FULL_PANEL else None
F30_ok = bool(F30a_ok and (F30b_ok is not False))
w('  F29 full_panel=%s ok=%s ; F30a(n=%d) dev=%.3e ok=%s ; F30b ok=%s (range=%s)' %
  (FULL_PANEL, F29_ok, _n_f30a, F30a_dev, F30a_ok, F30b_ok,
   ('%.3e' % F30b_range) if F30b_range is not None else 'NA'))
w('  F29 max|y1 - Phase12 recover| = %.3e (tol %.1e) ; ok=%s' % (F29_dev, ANCHOR_TOL, F29_dev <= ANCHOR_TOL))
w('  F30 max|y01 - 1| = %.3e (tol %.1e) ; ok=%s' % (F30_dev, ANCHOR_TOL, F30_dev <= ANCHOR_TOL))
sys.stdout.flush()

# ================================================================ A3a L6 位置曲线
w('')
w('--- A3a L6 位置曲线（mask={0} 与 {1}，dense 网格）---')
A3A = {}
if SMOKE:
    A3A['0'] = dose_mask(6, _A1_AL, _A1_PAIRS, (0,), per_pair=False)
    A3A['1'] = dose_mask(6, _A1_AL, _A1_PAIRS, (1,), per_pair=False)
else:
    A3A['0'] = dose_mask(6, AL_DEN, PAIRS_D, (0,), per_pair=False)
    A3A['1'] = dose_mask(6, AL_DEN, PAIRS_D, (1,), per_pair=False)
for tag in ('0', '1'):
    xs = [r['alpha'] for r in A3A[tag]]
    ys = [r['dDonor'] / FULL_SWAP for r in A3A[tag]]
    A3A[tag + '_xhalf'] = cross_alpha(xs, ys, XHF)
    A3A[tag + '_J'] = J_only(xs, ys)
    A3A[tag + '_y'] = ys
w('  mask={0}: xhalf=%.6f J=%.4f y(1)=%.6f' %
  (A3A['0_xhalf'] if A3A['0_xhalf'] is not None else float('nan'), A3A['0_J'], A3A['0_y'][-1]))
w('  mask={1}: xhalf=%.6f J=%.4f y(1)=%.6f' %
  (A3A['1_xhalf'] if A3A['1_xhalf'] is not None else float('nan'), A3A['1_J'], A3A['1_y'][-1]))
sys.stdout.flush()

# ================================================================ A2 readout R
w('')
w('--- A2 R 位点全位点代换 ---')
rowsR = dose_mask(R_SITE, (AL_LEG if not SMOKE else AL_LEG[:3] + [1.0]), _A1_PAIRS, (0, 1), per_pair=False)
xsR = [r['alpha'] for r in rowsR]; ysR = [r['dDonor'] / FULL_SWAP for r in rowsR]
A2 = dict(x=xsR, y=ysR, xhalf=cross_alpha(xsR, ysR, XHF), J=J_only(xsR, ysR),
          recover=ysR[-1], pert_rel=[r['pert_rel'] for r in rowsR])
w('  R: xhalf=%s J=%.4f recover=%.9f' %
  (('%.6f' % A2['xhalf']) if A2['xhalf'] is not None else 'None', A2['J'], A2['recover']))


# ---- R 的单点锚（mask={1}）
rowsR1 = dose_mask(R_SITE, [1.0], _A1_PAIRS, (1,), per_pair=False)
A2['recover_mask1'] = float(rowsR1[0]['dDonor']) / FULL_SWAP
w('  R: recover(mask={1}) = %.12f ; Phase12 profile_R_swap.y[-1] = %.12f' %
  (A2['recover_mask1'], R12['profile_R_swap']['y'][-1]))
sys.stdout.flush()

# ================================================================ A4 confirmation
w('')
w('--- A4 确认集（mask={0,1}，4 位点 x %d alpha x %d 对）---' % (len(AL_CONF), len(CONF_D)))
A4 = {}
if CONF_D:
    for s in CONF_SITES:
        rows = dose_mask(s, AL_CONF, CONF_D, (0, 1), per_pair=True)
        xs = [r['alpha'] for r in rows]; ys = [r['dDonor'] / FULL_SWAP for r in rows]
        A4[str(s)] = dict(x=xs, y=ys, xhalf=cross_alpha(xs, ys, XHF), J=J_only(xs, ys),
                          per_pair=[list(map(float, r['per_pair'])) for r in rows],
                          order=rows[0]['order'])
        w('  L%-3d xhalf=%-10s J=%-9.4f y(1)=%.6f' % (
            s, ('%.6f' % A4[str(s)]['xhalf']) if A4[str(s)]['xhalf'] is not None else 'None',
            A4[str(s)]['J'], ys[-1]))
else:
    w('  (SMOKE: skipped)')
sys.stdout.flush()

# ================================================================ A5 floor
w('')
w('--- A5 随机 5 维方向地板（mask={0,1}，alpha=1）---')
A5 = {}
if FLOOR_SITES and not SMOKE:
    rng = np.random.default_rng(SEED + 7)
    for s in FLOOR_SITES:
        vals = []
        for (rw, rs, dw, ds, sw) in PAIRS_D:
            V, B = VEC[rw], BASE[rw]
            src = srcs_of(V, s); dif = diffs_of(V, s)
            nd = float(np.mean([np.linalg.norm(dif[p]) for p in (0, 1)]))
            g = rng.normal(size=HID).astype(np.float32)
            xi = proj(g.astype(np.float64).astype(np.float32), AO).astype(np.float32)
            xi = xi / max(float(np.linalg.norm(xi)), 1e-9) * nd
            vecs = {p: (src[p] + xi).astype(np.float32) for p in (0, 1)}
            lg = fwd_patch(TMPL % rw, s, vecs)
            x1, _, _ = _acc(lg, B, rw, rs, dw, ds)
            vals.append(x1)
        A5[str(s)] = dict(y=float(np.mean(vals)) / FULL_SWAP, n=len(vals))
        w('  L%-3d floor y = %.6f' % (s, A5[str(s)]['y']))
    F_floor_max = max(v['y'] for v in A5.values())
    w('  max floor = %.6f' % F_floor_max)
else:
    w('  (SMOKE: skipped)')
sys.stdout.flush()

# ================================================================ A8 逐层累积层代换（amend2）
w('')
w('--- A8 逐层累积层代换（amend2；support order i，alpha legacy）---')
_A8_SITES = SITES if not SMOKE else SITES[:3]
_A8_AL = AL_LEG if not SMOKE else [0.0, 0.2, 0.5, 1.0]
_A8_PR = PAIRS_D if not SMOKE else PAIRS_D[:6]
t_a8 = time.time()
A8 = {}
for i, ell in enumerate(_A8_SITES):
    sup = [j for j in _A8_SITES if j <= ell]
    A8[str(i)] = dose_cumlayer(ell, sup, _A8_AL, _A8_PR, per_pair=True)
w('  A8 done %.1fs' % (time.time() - t_a8))
A8_CURVES, A8_XH, A8_J, A8_PER = {}, {}, {}, {}
for i in range(len(_A8_SITES)):
    xs, ys = curve_from_rows(A8[str(i)])
    A8_CURVES[str(i)] = dict(x=xs, y=ys, support=A8[str(i)][0]['support'],
                             pert_rel=[float(r['pert_rel']) for r in A8[str(i)]])
    A8_PER[str(i)] = [list(map(float, r['per_pair'])) for r in A8[str(i)]]
    A8_XH[str(i)] = cross_alpha(xs, ys, XHF)
    A8_J[str(i)] = J_only(xs, ys)
w('%-4s %-24s %-11s %-11s %-11s' % ('i', 'support', 'xhalf_A8', 'J_A8', 'y(i,1)'))
for i in range(len(_A8_SITES)):
    sup = A8_CURVES[str(i)]['support']
    w('%-4d %-24s %-11s %-11.4f %-11.6f' % (
        i, 'L%d..L%d (n=%d)' % (sup[0], sup[-1], len(sup)),
        ('%.6f' % A8_XH[str(i)]) if A8_XH[str(i)] is not None else 'None',
        A8_J[str(i)], A8_CURVES[str(i)]['y'][-1]))
sys.stdout.flush()

# ================================================================ A6 集中度（双坐标）+ bootstrap
w('')
w('--- A6 双坐标集中度（铁律 t）：A1（位置前缀）与 A8（逐层累积）各一份 ---')
n_xh_ok = int(np.sum([A1_XH[str(s)] is not None for s in _A1_SITES]))
F32_ok = (n_xh_ok == len(_A1_SITES))
F33_ok = all(np.isfinite(A1_J[str(s)]) for s in _A1_SITES) and (not SMOKE)


def _hist(v):
    v = np.asarray(v)[np.asarray(v) >= 0]
    if len(v) == 0:
        return {}
    u, c = np.unique(v, return_counts=True)
    return {str(int(k)): int(n) for k, n in zip(u, c)}


def full_concentration(per_list, alpha_grid, sites, tag, do_boot=True):
    """点估计 + bootstrap 双坐标集中度 + 置换零假设 + 位点间配对 Δ（口径逐字沿用 Phase 13）。"""
    PM = np.stack([np.array(p, float) for p in per_list], 0)
    nS = PM.shape[0]; nP = PM.shape[2]
    xs = np.array(alpha_grid, float)
    Y = PM.mean(axis=2) / FULL_SWAP
    xh = np.array([cross_alpha(xs, Y[i], XHF) for i in range(nS)], float)
    Jv = np.array([J_only(xs, Y[i]) for i in range(nS)], float)
    share_x, axw_x, jx = conc_hat(xh)
    share_j, axw_j, jj = conc_hat(Jv)
    o = dict(tag=tag, sites=list(sites), alpha_grid=list(alpha_grid),
             xhalf=[float(v) for v in xh], J=[float(v) for v in Jv],
             top3_x=share_x, argmax_w_x=axw_x, jumps_x=jx,
             top3_j=share_j, argmax_w_j=axw_j, jumps_j=jj,
             range_x=(float(xh[np.isfinite(xh)].max() - xh[np.isfinite(xh)].min())
                      if np.isfinite(xh).any() else float('nan')),
             range_j=(float(Jv[np.isfinite(Jv)].max() - Jv[np.isfinite(Jv)].min())
                      if np.isfinite(Jv).any() else float('nan')),
             n_finite_x=int(np.isfinite(xh).sum()), n_finite_j=int(np.isfinite(Jv).sum()),
             win_sem_x=(None if axw_x is None else dict(w=axw_x, a=sites[axw_x], b=sites[axw_x + 3])),
             win_sem_j=(None if axw_j is None else dict(w=axw_j, a=sites[axw_j], b=sites[axw_j + 3])))
    if (not do_boot) or nS < 4 or nP < 8:
        o['bootstrap'] = dict(skipped=True)
        o['null'] = dict(skipped=True)
        o['paired'] = dict(skipped=True, N_dec_J=0, N_dec_X=0, rows=[], n_pairs=max(nS - 1, 0))
        return o
    BRNG = np.random.default_rng(SEED)
    xh_b = np.full((BS, nS), np.nan); J_b = np.full((BS, nS), np.nan)
    t3x_b = np.full(BS, np.nan); t3j_b = np.full(BS, np.nan)
    ax_b = np.full(BS, -1, int); aj_b = np.full(BS, -1, int)
    dJ_b = np.full((BS, nS - 1), np.nan); dX_b = np.full((BS, nS - 1), np.nan)
    t_b = time.time()
    for b in range(BS):
        idx = BRNG.integers(0, nP, nP)
        fs_b = float(FS_VEC[idx].mean())
        if abs(fs_b) < 1e-9:
            continue
        Yb = PM[:, :, idx].mean(axis=2) / fs_b
        for i in range(nS):
            xv = cross_alpha(xs, Yb[i], XHF)
            xh_b[b, i] = xv if xv is not None else np.nan
            J_b[b, i] = J_only(xs, Yb[i])
        dX_b[b] = xh_b[b, :-1] - xh_b[b, 1:]
        dJ_b[b] = J_b[b, :-1] - J_b[b, 1:]
        xr = xh_b[b]
        if np.all(np.isfinite(xr)):
            rr = float(xr.max() - xr.min()); jm = np.diff(xr)
            if rr > 1e-12:
                ws = [abs(float(np.sum(jm[j:j + W]))) for j in range(len(jm) - W + 1)]
                t3x_b[b] = max(ws) / rr; ax_b[b] = int(np.argmax(ws))
        jr = J_b[b]
        if np.all(np.isfinite(jr)):
            rr = float(jr.max() - jr.min()); jm = np.diff(jr)
            if rr > 1e-12:
                ws = [abs(float(np.sum(jm[j:j + W]))) for j in range(len(jm) - W + 1)]
                t3j_b[b] = max(ws) / rr; aj_b[b] = int(np.argmax(ws))
    fnx = t3x_b[np.isfinite(t3x_b)]; fnj = t3j_b[np.isfinite(t3j_b)]
    hx = _hist(ax_b); hj = _hist(aj_b)
    mx = int(max(hx, key=lambda k: hx[k])) if hx else -1
    mj = int(max(hj, key=lambda k: hj[k])) if hj else -1
    o['bootstrap'] = dict(
        BS=BS, secs=round(time.time() - t_b, 1),
        P_ge_060_x=(float(np.mean(fnx >= 0.60)) if len(fnx) else None),
        P_le_040_x=(float(np.mean(fnx <= 0.40)) if len(fnx) else None),
        P_ge_060_j=(float(np.mean(fnj >= 0.60)) if len(fnj) else None),
        P_le_040_j=(float(np.mean(fnj <= 0.40)) if len(fnj) else None),
        mode_x=mx, mode_j=mj,
        freq_x=((hx[str(mx)] / max(sum(hx.values()), 1)) if hx else None),
        freq_j=((hj[str(mj)] / max(sum(hj.values()), 1)) if hj else None),
        hist_x=hx, hist_j=hj, n_ok_x=int(len(fnx)), n_ok_j=int(len(fnj)),
        ci_top3_x=_ci(t3x_b), ci_top3_j=_ci(t3j_b))
    NBRNG = np.random.default_rng(SEED + 13)
    t3n_x = np.full(BP, np.nan); t3n_j = np.full(BP, np.nan)
    jxv = np.array(jx, float); jjv = np.array(jj, float)
    for b in range(BP):
        px = jxv[NBRNG.permutation(len(jxv))]; pj = jjv[NBRNG.permutation(len(jjv))]
        wx = [abs(float(np.sum(px[j:j + W]))) for j in range(len(px) - W + 1)]
        wj = [abs(float(np.sum(pj[j:j + W]))) for j in range(len(pj) - W + 1)]
        t3n_x[b] = max(wx) / o['range_x']; t3n_j[b] = max(wj) / o['range_j']
    nx95 = float(np.percentile(t3n_x, 95)); nj95 = float(np.percentile(t3n_j, 95))
    o['null'] = dict(BP=BP, null_x_95=nx95, null_j_95=nj95,
                     x_above_null=bool(share_x is not None and share_x >= nx95),
                     j_above_null=bool(share_j is not None and share_j >= nj95),
                     ci_x=_ci(t3n_x), ci_j=_ci(t3n_j))
    rows = []; ndJ = 0; ndX = 0
    for i in range(nS - 1):
        rj_ = _ci(dJ_b[:, i]); rx_ = _ci(dX_b[:, i])
        lj = ('DECISIVE_DOWN' if (rj_['lo'] is not None and rj_['lo'] > 0)
              else 'DECISIVE_UP' if (rj_['hi'] is not None and rj_['hi'] < 0) else 'TIE')
        lx = ('DECISIVE_DOWN' if (rx_['lo'] is not None and rx_['lo'] > 0)
              else 'DECISIVE_UP' if (rx_['hi'] is not None and rx_['hi'] < 0) else 'TIE')
        ndJ += int(lj != 'TIE'); ndX += int(lx != 'TIE')
        rows.append(dict(a=sites[i], b=sites[i + 1],
                         dJ=dict(obs=float(Jv[i] - Jv[i + 1]), **rj_), label_j=lj,
                         dX=dict(obs=float(xh[i] - xh[i + 1]), **rx_), label_x=lx))
    o['paired'] = dict(rows=rows, N_dec_J=int(ndJ), N_dec_X=int(ndX), n_pairs=int(nS - 1))
    return o


A1C = full_concentration([A1_PER[str(s)] for s in _A1_SITES], _A1_AL, _A1_SITES,
                         'A1_position_prefix', do_boot=(not SMOKE))
_A8i = list(range(len(_A8_SITES)))
A8C = full_concentration([A8_PER[str(i)] for i in _A8i], _A8_AL, _A8i,
                         'A8_cumulative_layer', do_boot=(not SMOKE))

for _C in (A1C, A8C):
    _bt = _C['bootstrap']
    w('')
    w('  [%s] top3_x = %s (argmax_w %s) ; top3_j = %s (argmax_w %s)' %
      (_C['tag'], _C['top3_x'], _C['argmax_w_x'], _C['top3_j'], _C['argmax_w_j']))
    if _C['win_sem_x']:
        w('    X 窗口: w=%d <=> %s -> %s' % (_C['win_sem_x']['w'], _C['win_sem_x']['a'], _C['win_sem_x']['b']))
    if _C['win_sem_j']:
        w('    J 窗口: w=%d <=> %s -> %s' % (_C['win_sem_j']['w'], _C['win_sem_j']['a'], _C['win_sem_j']['b']))
    w('    range_x = %.6f ; range_j = %.6f' % (_C['range_x'], _C['range_j']))
    if _bt.get('skipped'):
        w('    bootstrap: skipped')
    else:
        w('    P(>=.60) X=%.4f J=%.4f ; P(<=.40) X=%.4f J=%.4f ; n_ok %d/%d (%.1fs)' %
          (_bt['P_ge_060_x'], _bt['P_ge_060_j'], _bt['P_le_040_x'], _bt['P_le_040_j'],
           _bt['n_ok_x'], _bt['n_ok_j'], _bt['secs']))
        w('    mode X=%d (freq %.4f) hist %s' % (_bt['mode_x'], _bt['freq_x'], _bt['hist_x']))
        w('    mode J=%d (freq %.4f) hist %s' % (_bt['mode_j'], _bt['freq_j'], _bt['hist_j']))
        w('    null 95th: X=%.4f (above=%s) J=%.4f (above=%s)' %
          (_C['null']['null_x_95'], _C['null']['x_above_null'],
           _C['null']['null_j_95'], _C['null']['j_above_null']))
        w('    paired Δ: N_dec_J=%d/%d N_dec_X=%d/%d' %
          (_C['paired']['N_dec_J'], _C['paired']['n_pairs'],
           _C['paired']['N_dec_X'], _C['paired']['n_pairs']))

xh_arr = np.array(A1C['xhalf'], float)
J_arr = np.array(A1C['J'], float)
share_x = A1C['top3_x']; axw_x = A1C['argmax_w_x']; jumps_x = A1C['jumps_x']
share_j = A1C['top3_j']; axw_j = A1C['argmax_w_j']; jumps_j = A1C['jumps_j']
_b1 = A1C['bootstrap']
mode_x = _b1.get('mode_x', -1); mode_j = _b1.get('mode_j', -1)
freq_x = _b1.get('freq_x'); freq_j = _b1.get('freq_j')
P_ge_x = _b1.get('P_ge_060_x'); P_le_x = _b1.get('P_le_040_x')
P_ge_j = _b1.get('P_ge_060_j'); P_le_j = _b1.get('P_le_040_j')
null_x_95 = (A1C['null']['null_x_95'] if not A1C['null'].get('skipped') else float('nan'))
null_j_95 = (A1C['null']['null_j_95'] if not A1C['null'].get('skipped') else float('nan'))
A6 = dict(A1=A1C, A8=A8C)
A6b = A1C['bootstrap']
A6n = A1C['null']
A7p = A1C['paired']
sys.stdout.flush()

# ================================================================ A7 range / steepness
w('')
w('--- A7 网格与统计量替代 ---')
rag = {}
for tag, al in (('legacy', AL_LEG), ('dense', AL_DEN)):
    vals = []
    for s in _A1_SITES:
        xs = A1_CURVES[str(s)]['x']; ys = A1_CURVES[str(s)]['y']
        m = [i for i, a in enumerate(xs) if a in al]
        v = cross_alpha([xs[i] for i in m], [ys[i] for i in m], XHF)
        vals.append(v)
    ok = [v for v in vals if v is not None]
    rag[tag] = dict(n_ok=len(ok), min=float(min(ok)) if ok else None,
                    max=float(max(ok)) if ok else None,
                    range=float(max(ok) - min(ok)) if ok else None)
    w('  XH_RANGE_p(%s, %d pts) = %s  (min %s max %s, n_ok %d)' %
      (tag, len(al), ('%.6f' % rag[tag]['range']) if ok else 'None',
       ('%.6f' % rag[tag]['min']) if ok else '-', ('%.6f' % rag[tag]['max']) if ok else '-', len(ok)))
if not SMOKE:
    rho_J = spearman(J_arr, np.array([J_12[s] for s in _A1_SITES], float))
    rho_xh = spearman(xh_arr, np.array([XH_12[s] for s in _A1_SITES], float))
    w('  spearman(J_p, J_swap)   = %.4f' % rho_J)
    w('  spearman(xhalf_p, xhalf_12) = %.4f' % rho_xh)
else:
    rho_J = rho_xh = float('nan')
A7 = dict(range_grid=rag, rho_Jp_vs_Jswap=rho_J, rho_xhalfp_vs_xhalf12=rho_xh,
          J_iqr_sites={str(s): A1_Ji[str(s)] for s in _A1_SITES})
sys.stdout.flush()

# ================================================================ 判决
w('')
w('--- 判决 ---')
G0p = bool(F24_ok and F25_ok and F26_ok and F27_ok and F28_ok
           and (F29_ok is not False) and (F30_ok is not False) and F31_ok)
endpoint_dev = F30b_dev if F30b_dev is not None else float('nan')


def verdict_of(C):
    """同坐标判决表（6 行）+ 跨族迁移判据（4 行），逐条对齐 seal/amend2 的冻结文本。"""
    if not isinstance(C, dict) or C.get('skipped'):
        return dict(verdict_same_coordinate='SKIPPED', verdict_cross_family='SKIPPED')
    bts = C.get('bootstrap') or {}
    s_x = C['top3_x']; s_j = C['top3_j']; w_x = C['argmax_w_x']; w_j = C['argmax_w_j']
    pg_x = bts.get('P_ge_060_x'); pl_x = bts.get('P_le_040_x')
    pg_j = bts.get('P_ge_060_j'); pl_j = bts.get('P_le_040_j')
    cd = bool(w_x is not None and w_j is not None and w_x != w_j and abs(w_x - w_j) >= 3
              and s_x is not None and s_x >= 0.40 and s_j is not None and s_j >= 0.40)
    if pg_x is not None and pg_j is not None and pg_x >= 0.95 and pg_j >= 0.95:
        vs = 'CONCENTRATION_FEW_LAYER_ROBUST'
    elif pl_x is not None and pl_j is not None and pl_x >= 0.95 and pl_j >= 0.95:
        vs = 'CONCENTRATION_ACCUMULATE_ROBUST'
    elif cd:
        vs = 'CONCENTRATION_COORDINATE_DEPENDENT'
    else:
        vs = 'CONCENTRATION_UNDECIDED'
    if w_x is None or w_j is None:
        vf = 'TRANSFER_UNDECIDED'
    else:
        dx = abs(w_x - MODE_X_13); dj = abs(w_j - MODE_J_13)
        if dx <= 2 and dj <= 2:
            vf = 'FAMILY_TRANSFER_BOTH'
        elif dx <= 2:
            vf = 'FAMILY_TRANSFER_XHALF_ONLY'
        elif dj <= 2:
            vf = 'FAMILY_TRANSFER_J_ONLY'
        else:
            vf = 'FAMILY_NO_TRANSFER'
    return dict(verdict_same_coordinate=vs, verdict_cross_family=vf, coord_dep=bool(cd),
                share_x=s_x, share_j=s_j, argmax_w_x=w_x, argmax_w_j=w_j,
                P_ge_060_x=pg_x, P_le_040_x=pl_x, P_ge_060_j=pg_j, P_le_040_j=pl_j,
                d_x=(abs(w_x - MODE_X_13) if w_x is not None else None),
                d_j=(abs(w_j - MODE_J_13) if w_j is not None else None),
                mode_x=bts.get('mode_x'), mode_j=bts.get('mode_j'),
                freq_x=bts.get('freq_x'), freq_j=bts.get('freq_j'))


V_A8 = verdict_of(A8C)
V_A1 = verdict_of(A1C)
if not G0p:
    verdict_same = verdict_fam = 'DEVICE_ANCHOR_FAILED'
    verdict_same_A1 = verdict_fam_A1 = 'DEVICE_ANCHOR_FAILED'
else:
    verdict_same = V_A8['verdict_same_coordinate']      # 主第三口径 = A8 逐层累积
    verdict_fam = V_A8['verdict_cross_family']
    verdict_same_A1 = ('PREFIX_ENDPOINT_NOT_DEGENERATE' if (endpoint_dev >= 1e-9)
                       else V_A1['verdict_same_coordinate'])
    verdict_fam_A1 = V_A1['verdict_cross_family']
w('  G0p = %s' % G0p)
w('  端点/恒等: A1 面板级 max|y01-1| = %s ; A1 逐对恒等 max rel dev = %.3e (n=%d)' %
  (('%.3e' % endpoint_dev) if endpoint_dev == endpoint_dev else 'NA', F30a_dev, _n_f30a))
w('  [A8 逐层累积 · 主第三口径] %s | %s' % (verdict_same, verdict_fam))
w('     share_x=%s w_x=%s | share_j=%s w_j=%s | P(>=.60) X=%s J=%s' %
  (V_A8['share_x'], V_A8['argmax_w_x'], V_A8['share_j'], V_A8['argmax_w_j'],
   V_A8['P_ge_060_x'], V_A8['P_ge_060_j']))
w('     d_x = %s (靶 %d) ; d_j = %s (靶 %d)' % (V_A8['d_x'], MODE_X_13, V_A8['d_j'], MODE_J_13))
w('  [A1 位置前缀 · 阴性对照] %s | %s' % (verdict_same_A1, verdict_fam_A1))
w('     share_x=%s w_x=%s | share_j=%s w_j=%s' % (V_A1['share_x'], V_A1['argmax_w_x'],
                                                  V_A1['share_j'], V_A1['argmax_w_j']))

# 位置判决
pos_verdict = 'NA'
add_verdict = 'NA'
if len(rat):
    med = float(np.median(rat))
    if med <= 0.30:
        pos_verdict = 'POS_LAST_POSITION_DOMINANT'
    elif med <= 0.70:
        pos_verdict = 'POS_TWO_POSITION_COMPARABLE'
    else:
        pos_verdict = 'POS_FIRST_POSITION_DOMINANT'
    add_verdict = 'SUB_ADDITIVE' if float(np.mean(Sdd > 0)) >= 0.5 else 'SUPER_ADDITIVE'
w('  position verdict = %s ; median y0/y1 = %s' % (pos_verdict,
                                                   ('%.4f' % float(np.median(rat))) if len(rat) else 'NA'))
w('  additivity = %s ; frac(S>0) = %s' % (add_verdict,
                                          ('%.4f' % float(np.mean(Sdd > 0))) if len(rat) else 'NA'))

# ================================================================ 预测核对
pc = {}
if not SMOKE:
    pc['P1'] = dict(desc=S['pre_registered_predictions']['P1']['desc'],
                    got_mode=mode_x, got_freq=freq_x,
                    pass_=bool(mode_x in (13, 14) and freq_x >= 0.50))
    pc['P2'] = dict(desc=S['pre_registered_predictions']['P2']['desc'],
                    got_mode=mode_j, got_freq=freq_j,
                    pass_=bool(mode_j in (0, 1, 2) and freq_j >= 0.50))
    pc['P3'] = dict(desc=S['pre_registered_predictions']['P3']['desc'],
                    got=rho_J, pass_=bool(np.isfinite(rho_J) and rho_J >= 0.50))
    n_gt = int(np.sum(np.array([_y0[s] for s in sorted(_y0)]) > np.array([_y1[s] for s in sorted(_y0)])))
    n_pos = int(np.sum(np.array([_y0[s] for s in sorted(_y0)]) > 0))
    pc['P4'] = dict(desc=S['pre_registered_predictions']['P4']['desc'],
                    got_n_y0_gt_y1=n_gt, got_n_y0_positive=n_pos, n_sites=len(_y0),
                    pass_=bool(n_gt >= 15 and n_pos >= 12))
    nS_pos = int(np.sum(Sdd > 0))
    pc['P5'] = dict(desc=S['pre_registered_predictions']['P5']['desc'],
                    got_n_S_positive=nS_pos, n_sites=len(Sdd),
                    pass_=bool(nS_pos >= 15))
    _a8y = [A8_CURVES[str(i)]['y'][-1] for i in range(len(_A8_SITES))]
    _a8mono = all(_a8y[i + 1] >= _a8y[i] - 1e-12 for i in range(len(_a8y) - 1))
    pc['P6'] = dict(desc=AM2['fix_3_new_arm']['predictions_for_A8']['P6'],
                    got=[float(v) for v in _a8y], monotone=bool(_a8mono), y_at_i0=float(_a8y[0]),
                    pass_=bool(_a8mono and _a8y[0] >= 0.95))
    _b8 = A8C.get('bootstrap') or {}
    pc['P7'] = dict(desc=AM2['fix_3_new_arm']['predictions_for_A8']['P7'],
                    got_mode=_b8.get('mode_x'), got_freq=_b8.get('freq_x'),
                    pass_=bool(_b8.get('mode_x') in (13, 14) and (_b8.get('freq_x') or 0) >= 0.50))
    w('')
    w('--- 预注册预测 ---')
    for k in sorted(pc):
        w('  %s : pass=%s  %s' % (k, pc[k]['pass_'],
                                  json.dumps({kk: vv for kk, vv in pc[k].items()
                                              if kk not in ('desc', 'pass_')}, ensure_ascii=False)))
else:
    w('')
    w('--- 预注册预测（SMOKE 跳过）---')
sys.stdout.flush()

# ================================================================ floors
floors = {
    'F24': dict(ok=bool(F24_ok), rebuilt=FULL_SWAP, inherited=FULL_SWAP_INH),
    'F25': dict(ok=bool(F25_ok), dev=float(F25_dev), tol=N6_TOL, mean_n6=N6_MEAN, ref=N6_REF),
    'F26': dict(ok=bool(F26_ok), dev=float(F26_dev), sing=[float(x) for x in SV6]),
    'F27': dict(ok=bool(F27_ok), distinct_T=T_ALL),
    'F28': dict(ok=bool(F28_ok), dev=float(F28_dev), detail=F28_detail),
    'F29': dict(ok=(None if F29_ok is None else bool(F29_ok)), dev=float(F29_dev),
                full_panel=bool(FULL_PANEL), note='y1(ell) == Phase12 recover(ell)（面板级，SMOKE 跳过）'),
    'F30': dict(ok=(None if F30_ok is None else bool(F30_ok)),
                F30a=dict(ok=bool(F30a_ok), dev=float(F30a_dev), n_pairs= int(_n_f30a),
                          note='逐对恒等式 per_pair(alpha=1) == FULL_SWAP_pairs[rw]（与子集无关）'),
                F30b=dict(ok=F30b_ok, range=F30b_range, dev=F30b_dev, full_panel=bool(FULL_PANEL),
                          note='面板级：y01(ell) 在 18 位点为常数且 == 1.0'),
                note='amend2 修正后的两条并列判据'),
    'F31': dict(ok=bool(F31_ok), bad=bad_base),
    'F32': dict(ok=bool(F32_ok), n_ok=int(np.sum([A1_XH[str(s)] is not None for s in _A1_SITES])), n=len(_A1_SITES)),
    'F33': dict(ok=bool(F33_ok), skipped=bool(SMOKE), note='J_only 全位点有限（SMOKE 网格过疏会出 inf）'),
    'F35': dict(ok=bool(F35_ok), dev=float(qell_dev)),
    'G0p': dict(ok=bool(G0p)),
}
w('')
w('--- floors ---')
for k in sorted(floors):
    w('  %-4s ok=%s %s' % (k, floors[k].get('ok'), json.dumps({kk: vv for kk, vv in floors[k].items()
                                                                if kk != 'ok'}, ensure_ascii=False)[:200]))

# ================================================================ 落盘
elapsed = time.time() - t0
res = {
    'phase': 14,
    'name': S['name'],
    'model': MODEL,
    'smoke': SMOKE,
    'elapsed_s': round(elapsed, 1),
    'seal_sha8': sha(SEAL)[:8], 'seal_sha256': sha(SEAL), 'exec_sha8': sha(EXEC)[:8],
    'amend1': dict(sha8=sha(AMEND)[:8], sha256=sha(AMEND), kind=AM['kind'],
                   trigger=AM['trigger'], corrected=AM['corrected_fields']),
    'inherits': dict(phase12_result_sha8=INH['phase12_result_sha8'],
                     phase13_result_sha8=INH['phase13_result_sha8'],
                     FULL_SWAP=FULL_SWAP_INH, mean_n6_ref_phase9=N6_REF,
                     MODE_X_13=MODE_X_13, MODE_J_13=MODE_J_13, XH_RANGE_12=XH_RANGE_12),
    'amend2': dict(sha8=sha(AMEND2)[:8], sha256=sha(AMEND2), kind=AM2['kind'],
                   trigger=AM2['trigger']),
    'panel': dict(discovery=len(PAIRS_D), confirmation=len(CONF_D), usable_pairs=len(PAIRS)),
    'layers': dict(L=L, primary=PRIMARY, n_heads=NH, head_dim=HD,
                   n_kv_heads=int(getattr(CFG, 'num_key_value_heads', NH)),
                   o_proj_in=int(ATTN0.in_features), norm_type=type(NORM).__name__),
    'sites': dict(profile=_A1_SITES, readout=R_SITE, conf=CONF_SITES, floor=FLOOR_SITES),
    'dose_coord': dict(full_swap=FULL_SWAP, mean_n6=N6_MEAN, alpha_legacy=AL_LEG, alpha_dense=AL_DEN),
    'A0a_full_swap': dict(rebuilt=FULL_SWAP, inherited=FULL_SWAP_INH, bit_equal=bool(F24_ok)),
    'A0b_n6': dict(mean_n6=N6_MEAN, ref=N6_REF, dev=float(F25_dev), ok=bool(F25_ok)),
    'A0c_u6': dict(sing=[float(x) for x in SV6], dev=float(F26_dev), ok=bool(F26_ok)),
    'A0d_noop': dict(dev=float(F28_dev), detail=F28_detail, ok=bool(F28_ok)),
    'A0e_tokenizer': dict(distinct_T=T_ALL, ok=bool(F27_ok)),
    'A1_curves': A1_CURVES,
    'A1_xhalf': A1_XH, 'A1_J': A1_J, 'A1_Jiqr': A1_Ji, 'A1_logistic': A1_LOG,
    'A1_perpair': A1_PER,
    'A2_readout': A2,
    'A3a_position_curves': A3A,
    'A3b_position_endpoints': dict(by_site=A3B, y0={str(k): v for k, v in _y0.items()},
                                   y1={str(k): v for k, v in _y1.items()},
                                   y01={str(k): v for k, v in _y01.items()}),
    'A3_position_summary': dict(
        y0={str(k): _y0[k] for k in sorted(_y0)}, y1={str(k): _y1[k] for k in sorted(_y1)},
        y01={str(k): _y01[k] for k in sorted(_y01)},
        ratio_y0_y1={str(k): (_y0[k] / _y1[k] if _y1[k] > 1e-9 else None) for k in sorted(_y0)},
        S={str(k): (_y0[k] + _y1[k] - _y01[k]) for k in sorted(_y0)},
        median_ratio=float(np.median(rat)) if len(rat) else None,
        n_y0_gt_y1=int(np.sum(np.array([_y0[s] for s in sorted(_y0)]) >
                              np.array([_y1[s] for s in sorted(_y0)]))) if _y0 else None,
        n_y0_positive=int(np.sum(np.array([_y0[s] for s in sorted(_y0)]) > 0)) if _y0 else None,
        n_S_positive=int(np.sum(Sdd > 0)) if len(Sdd) else None,
        verdict_position=pos_verdict, verdict_additivity=add_verdict),
    'A4_confirmation': A4,
    'A5_floor': A5,
    'A8_curves': A8_CURVES, 'A8_xhalf': A8_XH, 'A8_J': A8_J, 'A8_perpair': A8_PER,
    'A8_verdict': dict(V_A8=V_A8, V_A1=V_A1, endpoint_dev=(float(endpoint_dev) if endpoint_dev == endpoint_dev else None),
                       F30a_dev=float(F30a_dev), G0p=bool(G0p)),
    'A8_predictions': {k: pc[k] for k in ('P6', 'P7') if k in pc},
    'A6_concentration': A6,
    'A6_bootstrap_bands': A6b,
    'A6_null': A6n,
    'A7_range_grid': rag,
    'A7_steepness_alt': dict(rho_Jp_vs_Jswap=rho_J, rho_xhalfp_vs_xhalf12=rho_xh,
                             J_iqr={str(s): A1_Ji[str(s)] for s in _A1_SITES},
                             J_main={str(s): A1_J[str(s)] for s in _A1_SITES}),
    'A7_paired_delta': A7p,
    'predictions_check': pc,
    'verdict': dict(G0p=bool(G0p),
                    endpoint_dev=(float(endpoint_dev) if endpoint_dev == endpoint_dev else None),
                    verdict_same_coordinate=verdict_same,
                    verdict_cross_family=verdict_fam,
                    verdict_same_coordinate_A1=verdict_same_A1,
                    verdict_cross_family_A1=verdict_fam_A1,
                    primary_third_family='A8_cumulative_layer',
                    V_A8=V_A8, V_A1=V_A1,
                    verdict_position=pos_verdict, verdict_additivity=add_verdict),
    'floors': floors,
    'extra': dict(null_x_95=null_x_95, null_j_95=null_j_95,
                  null_x_above=bool(share_x is not None and share_x >= null_x_95),
                  null_j_above=bool(share_j is not None and share_j >= null_j_95),
                  XH_RANGE_p_legacy=rag['legacy']['range'], XH_RANGE_p_dense=rag['dense']['range'],
                  XH_RANGE_12=XH_RANGE_12, pertlim=PERT_LIM, phaselabel='N2h1-alpha-7',
                  amend2_sha8=sha(AMEND2)[:8],
                  A8_endpoint_curve_present=True,
                  F30a_pairs_checked=int(_n_f30a), FULL_PANEL=bool(FULL_PANEL)),
}

with io.open(RESULT, 'w', encoding='utf-8', newline='\n') as f:
    json.dump(res, f, ensure_ascii=False)
w('')
w('WROTE %s (%d B)' % (RESULT, os.path.getsize(RESULT)))
w('RESULT sha8 %s' % sha(RESULT)[:8])
ios = io.open(REPORT, 'w', encoding='utf-8', newline='\n')
ios.write('\n'.join(lines) + '\n')
ios.close()
w('WROTE %s (%d B)' % (REPORT, os.path.getsize(REPORT)))
w('REPORT sha8 %s' % sha(REPORT)[:8])
