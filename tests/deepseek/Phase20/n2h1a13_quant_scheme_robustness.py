# -*- coding: utf-8 -*-
"""
Phase 20 (N2h1-alpha-13) 主脚本：行为量与写入窗剖面的**跨精度稳健性**（nf4 vs bf16）。

动机（P19 §11 写死的死线）：
  P19 只把跨精度检验做到**向量侧**（w_l 谱与 com_V）。P18 的结论（行为质心 com_B 比向量质心浅、
  MLP 行为主导、spearman(w_all,b_all)>0）与 P16 的结论（com_layer(x)/com_layer(J)）全部在 nf4 口径下，
  从未在 bf16 下复算。本 Phase 把**行为侧**补齐。

本 Phase 的**唯一自变量 = 数值精度**（nf4 vs bf16）。其余逐字继承：
  - 模板 / 6 类词 / 41 实例 / 24 discovery 配对 / 17 confirmation 配对  ← Phase 16/17/18
  - U_l = 全 41 实例按类平均 -> 类别质心差 SVD，秩 = n_classes-1 = 5      ← Phase 16/17/18
  - 位点 = 1..L-2（与 P17 w_all 索引对齐）；REACH 域取 P16 冻结值（跨口径不变，用于同域配对）
  - nb 取 P17 冻结值；L*_own 取 P16 冻结值；ALPHAS / XHF / PROFILE 取 P16 冻结值

两个面板（同一个装置、同一次加载、同一批 capture）：
  [B] 行为预算（P18 口径）：
        b_{c,l} = mean_pairs [ score_of(h_l^R + P_{U_l}(dv_c)) - BASE[rw].sd0 ]
        c in INC_ALL / INC_MLP / INC_ATTN / INC_TOP1 / CUM_ALL（discovery）
        c in INC_ALL / INC_MLP / INC_ATTN（confirmation）
        附带 w_l 谱：w_c,l = mean_pairs ||P_{U_l}(dv_c)||（P17 口径，零额外前向）
  [P] 写入窗剖面（P16 口径）：
        dv = HH[l+1]^D - HH[l+1]^R（**原始**，不投影）；注入 h_l^R + alpha*dv；
        Y(l,alpha) = mean_disc dDonor / FULL_SWAP -> xhalf(l)=cross_alpha(.,0.5), J(l)=J_only(.)
        -> com_layer(x), com_layer(J), span3, rho(l)=Y(l,1), REACH 重算

用法：
  PROBE=1 python ...                          (A0 双口径、缩幅网格；只写 _probe20_*)
  SMOKE=1 python ...                          (A0_nf4、更小网格)
  SPLIT_PARTIAL=1 ARMS=A0_nf4 python ...      (逐臂进程隔离)
  MERGE=1 python ...                          (合并 -> result_phase20.json)
"""
import os
import sys
import io
import json
import time
import hashlib
import gc
import traceback

import numpy as np
import torch

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P20 = os.path.join(ROOT, 'tests', 'deepseek', 'Phase20')
P20T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase20')
P18T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase18')
P17T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase17')
P16T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase16')
EXECP = os.path.join(P20T, 'execution_phase20.json')
SEALP = os.path.join(P20T, 'N2h1a13_design_seal.json')
SMOKE = os.environ.get('SMOKE', '0') == '1'
PROBE = os.environ.get('PROBE', '0') == '1'
ARMS_SEL = os.environ.get('ARMS', '')
SPLIT_PARTIAL = os.environ.get('SPLIT_PARTIAL', '0') == '1'
MERGE = os.environ.get('MERGE', '0') == '1'
os.makedirs(P20T, exist_ok=True)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


EX = json.load(io.open(EXECP, encoding='utf-8'))
assert sha(SEALP) == EX['seal_sha256'], 'DRIFT: seal sha != execution.seal_sha256'

_ab18 = open(os.path.join(ROOT, EX['anchor_result_p18_path']), 'rb').read()
assert hashlib.sha256(_ab18).hexdigest() == EX['anchor_result_p18_sha256'], 'DRIFT: P18 result 锚漂移'
A18 = json.loads(_ab18.decode('utf-8'))
_ab16 = open(os.path.join(ROOT, EX['anchor_result_p16_path']), 'rb').read()
assert hashlib.sha256(_ab16).hexdigest() == EX['anchor_result_p16_sha256'], 'DRIFT: P16 result 锚漂移'
A16 = json.loads(_ab16.decode('utf-8'))
_ab17 = open(os.path.join(ROOT, EX['anchor_result_p17_path']), 'rb').read()
assert hashlib.sha256(_ab17).hexdigest() == EX['anchor_result_p17_sha256'], 'DRIFT: P17 result 锚漂移'
A17 = json.loads(_ab17.decode('utf-8'))

TMPL = EX['template']
SUPS = list(EX['classes'])
INST_ALL = [tuple(x) for x in EX['instances_all']]
PAIRS_ALL = [tuple(x) for x in EX['pairs_all']]
DISC = [tuple(x) for x in EX['discovery']]
CONF = [tuple(x) for x in EX['confirmation']]
DISC_W = set(x[0] for x in DISC)
CONF_W = set(x[0] for x in CONF)
QUANT_NF4 = EX['quant_nf4']
QUANT_BF16 = EX['quant_bf16']
COMPONENTS = list(EX['components'])
COMPONENTS_CONF = list(EX['components_confirmation'])
PROFILE = [int(x) for x in EX['profile_sites']]
PROFILE_LEGACY = [int(x) for x in EX['profile_sites_legacy']]
ALPHAS = [float(x) for x in EX['alphas']]
XHF = float(EX['xh_frac'])
FL = EX['floors']
BP = int(EX['bootstrap']['BP'])
SEEDS = dict(EX['bootstrap']['seeds'])
ARM_ORDER = list(EX['arm_order'])
ARMS_CFG = EX['arms']
NBW = int(EX['neighbourhood_width'])
ANCH = EX['anchors']

FULL_SCALE = (not SMOKE) and (not PROBE)
if SMOKE:
    # SMOKE 只求「跑通」：同时缩配对与网格
    BP = 200
    PAIRS_ALL = PAIRS_ALL[:8]
    _keep = []
    for _p in PAIRS_ALL:
        for _x in (_p[0], _p[2]):
            if _x not in _keep:
                _keep.append(_x)
    _sel = [t for t in INST_ALL if t[0] in _keep]
    INST_ALL = _sel if len(_sel) >= 4 else INST_ALL[:8]
    DISC = [d for d in DISC if d[0] in set(x[0] for x in INST_ALL)]
    CONF = [c for c in CONF if c[0] in set(x[0] for x in INST_ALL)]
    PROFILE = [1, 2, 3, 4, 5, 6, 8, 10, 12]
    ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
if PROBE:
    # 可行性探针：**保留全量配对与实例**（U_l 秩 = 5 才成立），只缩小网格与 BP
    BP = 200
    PROFILE = [1, 2, 3, 4, 5] + list(range(6, 35, 2))
    ALPHAS = [0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0]

_log = []


def w(s=''):
    _log.append(str(s))
    print(s)
    sys.stdout.flush()


def F3(v, nd=3):
    return ('%.' + str(nd) + 'f') % v if isinstance(v, (int, float)) and v is not None else str(v)


# ---------------------------------------------------------------- 统计工具（逐字继承 P16/P17/P18）
def stat_com_layer(jumps, sites):
    j = np.asarray(jumps, float)
    if len(j) == 0 or len(sites) != len(j) + 1:
        return None
    mid = (np.asarray(sites, float)[:-1] + np.asarray(sites, float)[1:]) / 2.0
    a = np.abs(j)
    den = float(a.sum())
    if not np.isfinite(den) or den <= 1e-12:
        return None
    return float((a * mid).sum() / den)


def stat_span_k(jumps, k=3):
    j = np.asarray(jumps, float)
    n = len(j)
    if n < k or not np.isfinite(j).all():
        return None
    idx = np.argsort(-np.abs(j))[:k]
    return float(idx.max() - idx.min()) / max(n - 1, 1)


def com_of_mass(mass_by_site, sites):
    """区间求和质心（与 P16/P17/P18 逐字节同口径）。"""
    s = np.asarray(sites, float)
    vals = np.asarray([float(sum(mass_by_site.get(int(l), 0.0)
                                for l in range(int(sites[j]), int(sites[j + 1]))))
                       for j in range(len(sites) - 1)], float)
    mid = (s[:-1] + s[1:]) / 2.0
    den = float(vals.sum())
    if not np.isfinite(den) or den <= 1e-12:
        return None, None
    return float((vals * mid).sum() / den), vals


def perm_null_com(mass_by_site, sites, rng_obj, n_bp):
    obs, vals = com_of_mass(mass_by_site, sites)
    if vals is None:
        return dict(BP=n_bp, n_ok=0, obs_com=None, com_p5=None, com_p95=None,
                    com_tail=None, reason='empty_or_zero_mass')
    n = len(vals)
    s = np.asarray(sites, float)
    mid = (s[:-1] + s[1:]) / 2.0
    out = np.full(n_bp, np.nan)
    for b in range(n_bp):
        p = vals[rng_obj.permutation(n)]
        den = float(p.sum())
        out[b] = float((p * mid).sum() / den) if den > 1e-12 else np.nan
    fin = out[np.isfinite(out)]
    if len(fin) == 0:
        return dict(BP=n_bp, n_ok=0, obs_com=obs, com_p5=None, com_p95=None,
                    com_tail=None, reason='all_nan')
    p5, p95 = float(np.percentile(fin, 5)), float(np.percentile(fin, 95))
    tail = ('low' if (obs is not None and obs <= p5) else
            'high' if (obs is not None and obs >= p95) else 'none')
    return dict(BP=n_bp, n_ok=int(len(fin)), obs_com=obs, com_p5=p5, com_p95=p95,
                com_tail=tail, reason=None)


def perm_null_share(b_mlp, b_attn, sites_reach, nb, rng_obj, n_bp):
    mi = {int(s): i for i, s in enumerate(sites_reach)}
    idx = [mi[l] for l in nb if l in mi]
    if not idx:
        return dict(BP=n_bp, n_ok=0, obs_share=None, share_p5=None, share_p95=None,
                    share_tail=None, reason='empty_nb')
    m = np.array([abs(b_mlp.get(int(l), 0.0)) for l in sites_reach], float)
    a = np.array([abs(b_attn.get(int(l), 0.0)) for l in sites_reach], float)
    den0 = m[idx].sum() + a[idx].sum()
    if den0 <= 1e-12:
        return dict(BP=n_bp, n_ok=0, obs_share=None, share_p5=None, share_p95=None,
                    share_tail=None, reason='zero_denominator')
    obs = float(m[idx].sum() / den0)
    n = len(sites_reach)
    out = np.full(n_bp, np.nan)
    for b in range(n_bp):
        mp = m[rng_obj.permutation(n)]
        d = mp[idx].sum() + a[idx].sum()
        out[b] = float(mp[idx].sum() / d) if d > 1e-12 else np.nan
    fin = out[np.isfinite(out)]
    if len(fin) == 0:
        return dict(BP=n_bp, n_ok=0, obs_share=obs, share_p5=None, share_p95=None,
                    share_tail=None, reason='all_nan')
    p5, p95 = float(np.percentile(fin, 5)), float(np.percentile(fin, 95))
    tail = ('low' if obs <= p5 else 'high' if obs >= p95 else 'none')
    return dict(BP=n_bp, n_ok=int(len(fin)), obs_share=obs, share_p5=p5, share_p95=p95,
                share_tail=tail, reason=None)


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = len(a)
    if n < 3:
        return None
    if float(np.std(a)) <= 1e-9 or float(np.std(b)) <= 1e-9:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    return float((ra * rb).sum() / den) if den > 1e-12 else None


def cross_alpha(xs, ys, frac):
    """逐字继承 Phase 12/16：第一个跨过 frac*max 的 alpha（线性内插）。"""
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    if len(xs) < 2:
        return None
    ym = float(np.max(ys))
    if not np.isfinite(ym) or abs(ym) < 1e-12:
        return None
    tgt = frac * ym
    for i in range(len(xs) - 1):
        if ys[i] < tgt <= ys[i + 1]:
            t = (tgt - ys[i]) / (ys[i + 1] - ys[i])
            return float(xs[i] + t * (xs[i + 1] - xs[i]))
    return None


def J_only(xs, ys):
    """峰值斜率 / 其余斜率中位数（逐字沿用 Phase 12/16）。"""
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    m = xs >= 0.01
    xs2, ys2 = xs[m], ys[m]
    if len(xs2) < 3:
        return float('nan')
    s = np.diff(ys2) / np.diff(xs2)
    k = int(np.argmax(s))
    rest = np.delete(s, k)
    smed = float(np.median(rest)) if len(rest) else 0.0
    return float(s[k] / smed) if smed > 1e-12 else float('inf')


# ---------------------------------------------------------------- 单臂
def run_arm(arm_id, acfg):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    scheme = acfg['quant']
    akey = acfg['anchor_key']
    rec = dict(arm=arm_id, role=acfg['role'], model=acfg['model'], scheme=scheme,
               offload=bool(acfg.get('offload')), anchor_key=akey, smoke=SMOKE, probe=PROBE)
    MDIR = os.path.join(ROOT, 'models', 'hf', acfg['dir'])

    def ids_of(s):
        return tok.encode(s, add_special_tokens=False)

    def attn_out_proj(layer):
        a = layer.self_attn
        for nm in ('o_proj', 'dense', 'out_proj'):
            if hasattr(a, nm):
                return getattr(a, nm), nm
        raise RuntimeError('no attn out proj')

    def mod_dev(m):
        # [E-offload, 承 P19] accelerate CPU-offload 用 meta 占位参数承载真实权重；
        # 真实执行设备在 module._hf_hook.execution_device。用参数 device 会触发
        # 'Cannot copy out of meta tensor'。
        h = getattr(m, '_hf_hook', None)
        ed = getattr(h, 'execution_device', None) if h is not None else None
        if ed is not None:
            return ed
        for p in m.parameters():
            return p.device
        return torch.device('cuda')

    tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)

    # --- F1b 类别 token 逐臂现场解析（禁止跨词表沿用硬编码 id）
    SUP_ID = {}
    for _wd in SUPS:
        _t = list(tok.encode(_wd, add_special_tokens=False))
        assert len(_t) == 1 and tok.decode([_t[0]]) == _wd, \
            'F1b 失败：类别词 %r 非单 token 或 decode 不可逆 %r' % (_wd, _t)
        SUP_ID[_wd] = int(_t[0])
    rec['sup_id_arm'] = dict(SUP_ID)
    rec['F1b_ok'] = True

    tl = {}
    for wd, sup in INST_ALL:
        tl.setdefault(len(ids_of(TMPL % wd)), []).append(wd)
    rec['token_len_hist'] = {str(k): len(v) for k, v in sorted(tl.items())}
    rec['T2_only'] = bool(sorted(tl.keys()) == [2])
    w('  F1 T=2: hist=%s only=%s | F1b sup_id=%s' % (rec['token_len_hist'], rec['T2_only'],
                                                    json.dumps(SUP_ID, ensure_ascii=False)))

    max_mem = {int(k) if str(k).isdigit() else k: v for k, v in QUANT_NF4['max_memory'].items()}
    t0 = time.time()
    if scheme == 'nf4':
        bnb = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_quant_type=QUANT_NF4['bnb_4bit_quant_type'],
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=bool(QUANT_NF4['bnb_4bit_use_double_quant']))
        model = AutoModelForCausalLM.from_pretrained(
            MDIR, quantization_config=bnb, trust_remote_code=True,
            attn_implementation=QUANT_NF4['attn_implementation'], low_cpu_mem_usage=True,
            device_map=QUANT_NF4['device_map'], max_memory=max_mem)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MDIR, dtype=torch.bfloat16, trust_remote_code=True,
            attn_implementation=QUANT_BF16['attn_implementation'], low_cpu_mem_usage=True,
            device_map=QUANT_BF16['device_map'], max_memory=max_mem)
    model.eval()
    rec['load_s'] = round(time.time() - t0, 1)
    w('  loaded %.1fs (%s)' % (rec['load_s'], scheme))

    layers = model.model.layers
    L = len(layers)
    CFG = model.config
    HID = int(CFG.hidden_size)
    NH = int(CFG.num_attention_heads)
    HDP = int(getattr(CFG, 'head_dim', HID // max(NH, 1)))
    OPR = [attn_out_proj(layers[l])[0] for l in range(L)]
    MLPS = [layers[l].mlp for l in range(L)]
    OIN = int(OPR[0].in_features)
    onm = attn_out_proj(layers[0])[1]
    INDEV = model.get_input_embeddings().weight.device
    rec['config_sha256'] = sha(os.path.join(MDIR, 'config.json'))
    dev_hist = {}
    for _, p in model.named_parameters():
        dev_hist[str(p.device)] = dev_hist.get(str(p.device), 0) + 1
    exp = acfg['expected']
    drift = []
    if L != exp['num_hidden_layers']:
        drift.append('L')
    if HID != exp['hidden_size']:
        drift.append('hid')
    if NH != exp['num_attention_heads']:
        drift.append('n_heads')
    if HDP != exp['head_dim']:
        drift.append('head_dim')
    if OIN != NH * HDP:
        drift.append('o_proj_in')
    rec['drift'] = drift
    rec['F4_dims_ok'] = (len(drift) == 0)
    rec['F5_o_proj_ok'] = (OIN == NH * HDP)
    rec['cfg'] = dict(L=L, hid=HID, n_heads=NH, head_dim=HDP, o_proj_in=OIN, o_proj_name=onm,
                      tie=bool(getattr(CFG, 'tie_word_embeddings', False)),
                      vocab=int(CFG.vocab_size), param_devices=dev_hist)
    w('  F4 维度: L=%d hid=%d heads=%d head_dim=%d o_proj_in=%d(%s) tie=%s ; drift=%s'
      % (L, HID, NH, HDP, OIN, onm, bool(getattr(CFG, 'tie_word_embeddings', False)), drift))
    assert rec['F5_o_proj_ok'], 'F5 失败：o_proj_in %d != NH*HD %d' % (OIN, NH * HDP)
    all_cuda = all(k.startswith('cuda') for k in dev_hist)
    rec['Q0_device'] = 'cuda' if all_cuda else ('OFFLOAD' if scheme == 'bf16' else 'MIXED')

    # --- 位点与冻结域
    ALL_SITES = list(range(1, L - 1))
    REACH_FROZEN = [int(x) for x in acfg['reach']]
    RE_L = [s for s in REACH_FROZEN if s in ALL_SITES]
    nb = [int(x) for x in acfg['nb']]
    nb = [s for s in nb if s in RE_L]
    L_STAR = int(acfg['L_star_own'])
    PRO = [s for s in PROFILE if s <= L - 2]
    assert nb, 'nb 为空 —— REACH/邻域锚有问题'
    w('  全域位点 n=%d ; REACH n=%d ; nb=%s ; PROFILE n=%d' % (len(ALL_SITES), len(RE_L), nb, len(PRO)))

    def ids_t(text):
        return torch.tensor([ids_of(text)], device=INDEV)

    n_fw = 0
    # --- E0 装置自检
    @torch.no_grad()
    def fwd_patch(text, site, vec):
        ii = ids_t(text)
        mod = layers[site]

        def hook(m, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            t = t.clone()
            t[0, -1, :] = torch.as_tensor(vec, device=t.device, dtype=t.dtype)
            return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t
        h = mod.register_forward_hook(hook)
        try:
            o = model(input_ids=ii)
        finally:
            h.remove()
        return o.logits[0, -1].float().detach().cpu().numpy()

    with torch.no_grad():
        lga = model(input_ids=ids_t(TMPL % '苹果')).logits[0, -1].float().detach().cpu().numpy()
        lgb = model(input_ids=ids_t(TMPL % '苹果')).logits[0, -1].float().detach().cpu().numpy()
        n_fw += 2
    determinism = float(np.max(np.abs(lga - lgb)))
    rng0 = np.random.default_rng(EX['bootstrap']['seed'])
    vec0 = (rng0.standard_normal(HID) * 0.1).astype(np.float32)
    site0 = L // 2
    lg_hook = fwd_patch(TMPL % '苹果', site0, torch.tensor(vec0, device=INDEV))
    n_fw += 1
    hook_effect = float(np.max(np.abs(lg_hook - lgb)))
    rec['E0_selfcheck'] = dict(determinism_maxdiff=determinism, hook_site=site0,
                              hook_effect_maxdiff=hook_effect, Q0_device=rec['Q0_device'],
                              F4_dims_ok=rec['F4_dims_ok'], F5_o_proj_ok=rec['F5_o_proj_ok'])
    w('  E0 自检: determinism=%.3e ; hook@L%d effect=%.3e ; device=%s'
      % (determinism, site0, hook_effect, rec['Q0_device']))

    # --- E1 capture
    @torch.no_grad()
    def capture(text):
        ii = ids_t(text)
        so, sm = {}, {}
        hs = []
        for l in range(L):
            def mk_o(l):
                def f(mod, args):
                    so[l] = args[0].detach()[0, -1, :].float().cpu().numpy().copy()
                return f

            def mk_m(l):
                def f(mod, inp, out):
                    t = out[0] if isinstance(out, tuple) else out
                    sm[l] = t.detach()[0, -1, :].float().cpu().numpy().copy()
                return f
            hs.append(OPR[l].register_forward_pre_hook(mk_o(l)))
            hs.append(MLPS[l].register_forward_hook(mk_m(l)))
        out = model(input_ids=ii, output_hidden_states=True)
        for h in hs:
            h.remove()
        HH = np.stack([x[0, -1].float().detach().cpu().numpy() for x in out.hidden_states], 0)
        return HH, so, sm, out.logits[0, -1].float().detach().cpu().numpy()

    t0 = time.time()
    CAP = {}
    for wd, sup in INST_ALL:
        CAP[wd] = capture(TMPL % wd)
    n_fw += len(INST_ALL)
    rec['E1_capture'] = dict(n=len(CAP), seconds=round(time.time() - t0, 1),
                             hidden_levels=int(CAP[INST_ALL[0][0]][0].shape[0]))
    w('  E1 capture %d 实例 / %.1fs (levels=%d)'
      % (len(CAP), rec['E1_capture']['seconds'], rec['E1_capture']['hidden_levels']))
    HH0 = CAP[INST_ALL[0][0]][0]
    assert HH0.shape[1] == HID and not np.isnan(HH0).any(), 'capture NaN or shape'
    assert CAP[INST_ALL[0][0]][1][L - 1].shape[0] == OIN

    # --- 逐头分块
    NH1 = NH + 1
    blocks = np.zeros((NH1, OIN), np.float32)

    @torch.no_grad()
    def head_blocks(opmod, v):
        blocks[:] = 0.0
        for h in range(NH):
            blocks[h, h * HDP:(h + 1) * HDP] = v[h * HDP:(h + 1) * HDP]
        blocks[NH] = v
        t = torch.tensor(blocks, device=mod_dev(opmod), dtype=torch.bfloat16)
        o = opmod(t).float().cpu().numpy()
        return o[:NH], o[NH]

    # --- E2 保真度门
    arch, blk = [], []
    for wd, sup in INST_ALL[:6]:
        HHs, O, M, _ = CAP[wd]
        for l in range(L - 1):
            lhs = HHs[l + 1] - HHs[l]
            with torch.no_grad():
                rhs = OPR[l](torch.tensor(O[l][None, :], device=mod_dev(OPR[l]),
                                          dtype=torch.bfloat16)).detach().float().cpu().numpy()[0] + M[l]
            arch.append(float(np.linalg.norm(lhs - rhs)) / max(float(np.linalg.norm(lhs)), 1e-9))
    for wd, sup in INST_ALL[:4]:
        _, O, M, _ = CAP[wd]
        for l in range(0, L - 1, max(L // 8, 1)):
            hb, full = head_blocks(OPR[l], O[l])
            blk.append(float(np.linalg.norm(hb.sum(0) - full)) / max(float(np.linalg.norm(full)), 1e-9))
    arch = np.array(arch); blk = np.array(blk)
    rec['E2_fidelity'] = dict(arch_max=float(arch.max()), arch_mean=float(arch.mean()),
                              arch_p99=float(np.percentile(arch, 99)),
                              blk_max=float(blk.max()), blk_mean=float(blk.mean()),
                              n_arch=int(len(arch)), n_blk=int(len(blk)))
    w('  E2 保真度: arch max=%.3e mean=%.3e | blocks max=%.3e mean=%.3e'
      % (arch.max(), arch.mean(), blk.max(), blk.mean()))

    # --- E3 U_l（全 41 实例按类平均 -> 类别质心差 SVD，秩 = n_classes-1）
    by = {}
    for wd, sup in INST_ALL:
        by.setdefault(sup, []).append(wd)
    AVAIL = [s for s in SUPS if s in by]
    assert len(AVAIL) >= 2, 'AVAIL<2 => U_l 不可估'
    rk = max(len(AVAIL) - 1, 1)
    U = {}
    for l in range(L):
        mus = np.stack([np.mean([CAP[wd][0][l + 1] for wd in by[s]], 0) for s in AVAIL], 0).astype(np.float64)
        Dm = mus - mus.mean(0, keepdims=True)
        _, sv, Vt = np.linalg.svd(Dm, full_matrices=False)
        U[l] = Vt[:rk].astype(np.float32)
    rec['E3_U'] = dict(rank=rk, n_classes=len(AVAIL), classes=AVAIL)
    w('  E3 U_l: rank=%d n_classes=%d' % (rk, len(AVAIL)))

    def proj(v, Ub):
        return (v @ Ub.T) @ Ub

    # --- E4 BASE / FULL_SWAP（同 P16/P18 口径）
    def score_of(v, sup, sid):
        v = v.copy(); v[sid] = -1e9
        own = float(v[SUP_ID[sup]])
        others = [float(v[SUP_ID[x]]) for x in SUPS if x != sup]
        return own - float(np.mean(others))

    disc_pairs = [p for p in PAIRS_ALL if p[0] in DISC_W and p[0] in CAP and p[2] in CAP]
    conf_pairs = [p for p in PAIRS_ALL if p[0] in CONF_W and p[0] in CAP and p[2] in CAP]
    BASE = {}
    for (rw, rs, dw, ds, sw) in disc_pairs + conf_pairs:
        if rw in BASE:
            continue
        lg = CAP[rw][3]
        BASE[rw] = dict(sr0=score_of(lg, rs, ids_of(rw)[0]), sd0=score_of(lg, ds, ids_of(dw)[0]))
    bad_base = [rw for rw, b in BASE.items() if not (b['sr0'] > 0)]
    rec['E4_base'] = dict(n=len(BASE), bad=bad_base, n_disc=len(disc_pairs), n_conf=len(conf_pairs))
    FS_PAIR = {}
    for (rw, rs, dw, ds, sw) in disc_pairs:
        FS_PAIR[rw] = float(score_of(CAP[dw][3], ds, ids_of(dw)[0]) - BASE[rw]['sd0'])
    full_swap = float(np.mean([FS_PAIR[p[0]] for p in disc_pairs]))
    rec['E4_full_swap'] = dict(FULL_SWAP=full_swap, n=len(disc_pairs))
    w('  E4 base n=%d bad=%s ; FULL_SWAP=%.6f (disc=%d conf=%d)'
      % (len(BASE), bad_base if bad_base else 'NONE', full_swap, len(disc_pairs), len(conf_pairs)))

    # ------------------------------------------------------------ Panel B：行为预算 + 向量谱
    def sweep(pairs, comps, sites):
        out = {}
        for cname in comps:
            b, rel, per, wv = {}, {}, {}, {}
            for s in sites:
                dd = 0.0
                pr, rl, pp = [], [], []
                for (rw, rs, dw, ds, sw) in pairs:
                    HR, OR_, MR = CAP[rw][0], CAP[rw][1], CAP[rw][2]
                    HD, OD, MD = CAP[dw][0], CAP[dw][1], CAP[dw][2]
                    Ub = U[s]
                    hbD, fullD = head_blocks(OPR[s], OD[s])
                    hbR, fullR = head_blocks(OPR[s], OR_[s])
                    d_attn = fullD - fullR
                    d_mlp = MD[s] - MR[s]
                    if cname == 'INC_ALL':
                        dv = d_attn + d_mlp
                    elif cname == 'INC_MLP':
                        dv = d_mlp
                    elif cname == 'INC_ATTN':
                        dv = d_attn
                    elif cname == 'INC_TOP1':
                        dd_h = hbD - hbR
                        pv_ = np.array([float(np.linalg.norm(proj(dd_h[h], Ub))) for h in range(NH)])
                        dv = dd_h[int(np.argmax(pv_))]
                    elif cname == 'CUM_ALL':
                        dv = HD[s + 1] - HR[s + 1]
                    else:
                        raise RuntimeError(cname)
                    pv = proj(dv.astype(np.float32), Ub)
                    h0 = HR[s + 1].astype(np.float32)
                    lg = fwd_patch(TMPL % rw, s, h0 + pv)
                    x1 = score_of(lg, ds, ids_of(dw)[0]) - BASE[rw]['sd0']
                    dd += x1; pp.append(float(x1))
                    pr.append(float(np.linalg.norm(pv)))
                    rl.append(float(np.linalg.norm(pv)) / max(float(np.linalg.norm(h0)), 1e-9))
                b[s] = dd / max(len(pairs), 1)
                rel[s] = float(np.mean(rl)); per[s] = pp
                wv[s] = float(np.mean(pr))
            out[cname] = b
            out.setdefault('_rel', {})[cname] = rel
            out.setdefault('_wvec', {})[cname] = wv
            out.setdefault('_per_pair', {})[cname] = per
        return out

    t0 = time.time()
    SW_D = sweep(disc_pairs, COMPONENTS, ALL_SITES)
    n_fw += len(disc_pairs) * len(COMPONENTS) * len(ALL_SITES)
    rec['E5_seconds'] = round(time.time() - t0, 1)
    w('  E5 [B] discovery 扫描 %d 组件 x %d 位点 x %d 对 / %.1fs'
      % (len(COMPONENTS), len(ALL_SITES), len(disc_pairs), rec['E5_seconds']))

    t0 = time.time()
    SW_C = sweep(conf_pairs, COMPONENTS_CONF, ALL_SITES)
    n_fw += len(conf_pairs) * len(COMPONENTS_CONF) * len(ALL_SITES)
    rec['E6_seconds'] = round(time.time() - t0, 1)
    w('  E6 [B] confirmation 扫描 %d 组件 x %d 位点 x %d 对 / %.1fs'
      % (len(COMPONENTS_CONF), len(ALL_SITES), len(conf_pairs), rec['E6_seconds']))

    def bmap(sw, c):
        return {int(l): float(sw[c][l]) for l in ALL_SITES}

    B = {c: bmap(SW_D, c) for c in COMPONENTS}
    BC = {c: bmap(SW_C, c) for c in COMPONENTS_CONF}
    WV = {c: {int(l): float(SW_D['_wvec'][c][l]) for l in ALL_SITES} for c in COMPONENTS}
    # P17 的 w 谱（现场读入，用于跨 Phase 对照 / spearman(w,b)）
    W17 = {k: {int(l): float(v) for l, v in enumerate(A17['arms'][akey]['E5_com_V'][k])}
           for k in ('w_all', 'w_mlp', 'w_attn')}

    def cen(c, sites, table=None):
        return com_of_mass({l: abs((table or B)[c][l]) for l in ALL_SITES}, sites)[0]

    com_B = {c: cen(c, RE_L) for c in COMPONENTS}
    com_B_full = {c: cen(c, ALL_SITES) for c in COMPONENTS}
    com_B_conf = {c: cen(c, RE_L, BC) for c in COMPONENTS_CONF}
    com_V_recomputed = com_of_mass({l: W17['w_all'][l] for l in ALL_SITES}, RE_L)[0]
    com_V_own = com_of_mass({l: WV['INC_ALL'][l] for l in ALL_SITES}, RE_L)[0]

    def sh(c_num, c_den, sites, table=None):
        t = table or B
        num = float(sum(abs(t[c_num][l]) for l in sites))
        den = float(sum(abs(t[c_den][l]) for l in sites))
        return (num / den) if den > 1e-12 else None

    share_mlp_beh_nb = sh('INC_MLP', 'INC_ALL', nb)
    share_attn_beh_nb = sh('INC_ATTN', 'INC_ALL', nb)
    share_top1_beh_nb = sh('INC_TOP1', 'INC_ALL', nb)
    share_mlp_beh_reach = sh('INC_MLP', 'INC_ALL', RE_L)
    v_m = float(sum(W17['w_mlp'][l] for l in nb)); v_all = float(sum(W17['w_all'][l] for l in nb))
    share_mlp_vec_nb = (v_m / v_all) if v_all > 1e-12 else None
    # 本臂自己的 w 谱份额（Panel W 同口径）
    wm_own = float(sum(WV['INC_MLP'][l] for l in nb)); wa_own = float(sum(WV['INC_ALL'][l] for l in nb))
    share_mlp_wown_nb = (wm_own / wa_own) if wa_own > 1e-12 else None

    def com_layer_of(c, table=None):
        t = table or B
        arr = np.array([t[c][l] for l in RE_L], float)
        return stat_com_layer(np.diff(arr), RE_L)

    comlayer_B_all = com_layer_of('INC_ALL')
    comlayer_B_mlp = com_layer_of('INC_MLP')
    comlayer_B_attn = com_layer_of('INC_ATTN')

    def sper_anchor(bcomp):
        """P18 口径：w 用 P17 冻结锚谱（键名 w_all/w_mlp/w_attn）。"""
        return spearman([W17['w_all'][l] for l in RE_L], [abs(B[bcomp][l]) for l in RE_L])

    def sper_own(cw, cb, table=None):
        """同口径：w 用**本臂自己重算**的谱（键名 = 组件名 INC_*）。"""
        t = table or B
        return spearman([WV[cw][l] for l in RE_L], [abs(t[cb][l]) for l in RE_L])

    sp_wall_ball = sper_anchor('INC_ALL')                  # P18 口径（w 用 P17 冻结锚）
    sp_wall_ball_own = sper_own('INC_ALL', 'INC_ALL')       # 同口径（w 用本臂自己重算）
    sp_wmlp_bmlp_own = sper_own('INC_MLP', 'INC_MLP')
    J_p16 = {int(s): float(v) for s, v in zip(A16['E4_summary'][akey]['sites'],
                                              A16['E4_summary'][akey]['J'])}
    xl = [l for l in RE_L if l in J_p16]
    sp_wall_J = spearman([W17['w_all'][l] for l in xl], [J_p16[l] for l in xl])

    # 行为谱的秩相关（同对象、跨口径用；此处存本臂谱）
    rlin = {int(l): (abs(B['INC_ALL'][l] - (B['INC_MLP'][l] + B['INC_ATTN'][l])) /
                     max(abs(B['INC_ALL'][l]), 1e-12)) for l in ALL_SITES}
    rlin_nb = float(np.mean([rlin[l] for l in nb])) if nb else None
    rlin_reach = float(np.mean([rlin[l] for l in RE_L])) if RE_L else None
    _rl = np.array([rlin[l] for l in ALL_SITES], float)
    _order = np.argsort(-_rl)
    rlin_argmax = int(ALL_SITES[int(_order[0])])
    rlin_second = float(_rl[_order[1]]) if len(_order) > 1 else None
    rlin_peak_ratio = (float(_rl[_order[0]] / rlin_second) if (rlin_second and rlin_second > 1e-12) else None)
    cum_bridge = B['CUM_ALL'].get(L_STAR)
    bridge_rel = (abs(cum_bridge - full_swap) / max(abs(full_swap), 1e-12)
                  if (cum_bridge is not None and np.isfinite(cum_bridge)) else None)

    # ------------------------------------------------ Panel P：写入窗 α 剖面（P16 口径）
    @torch.no_grad()
    def dose_swap(site, pairs):
        out = []
        for a in ALPHAS:
            dd = 0.0
            per, perts, n = [], [], 0
            for (rw, rs, dw, ds, sw) in pairs:
                HR = CAP[rw][0]; HD = CAP[dw][0]
                h0 = HR[site + 1].astype(np.float32)
                dv = (HD[site + 1] - HR[site + 1]).astype(np.float32)
                lg = fwd_patch(TMPL % rw, site, h0 + a * dv)
                x1 = score_of(lg, ds, ids_of(dw)[0]) - BASE[rw]['sd0']
                dd += x1; n += 1
                per.append(float(x1))
                perts.append(float(a) * float(np.linalg.norm(dv)) / max(float(np.linalg.norm(h0)), 1e-9))
            out.append(dict(alpha=float(a), dDonor=dd / max(n, 1), n=n,
                            pert_rel=float(np.mean(perts)), per_pair=per))
        return out

    t0 = time.time()
    E4P = {}
    for s in PRO:
        E4P[str(s)] = dose_swap(s, disc_pairs)
    n_fw += len(PRO) * len(ALPHAS) * len(disc_pairs)
    rec['E7P_seconds'] = round(time.time() - t0, 1)
    w('  E7 [P] α 剖面 %d 位点 x %d α x %d 对 / %.1fs'
      % (len(PRO), len(ALPHAS), len(disc_pairs), rec['E7P_seconds']))

    PM = np.stack([np.array([r['per_pair'] for r in E4P[str(s)]], float) for s in PRO], 0)  # [nS,nA,nP]
    Y = PM.mean(axis=2) / full_swap
    xh = np.array([cross_alpha(ALPHAS, Y[i], XHF) for i in range(len(PRO))], dtype=float)
    xh = np.where(np.isfinite(xh), xh, np.nan)
    Jv = np.array([J_only(ALPHAS, Y[i]) for i in range(len(PRO))], dtype=float)
    a1i = list(ALPHAS).index(1.0) if 1.0 in ALPHAS else None
    rho = (Y[:, a1i] if a1i is not None else np.full(len(PRO), np.nan))
    UNR = float(FL['UNREACH_y'])
    REACH_own = [int(sv) for i, sv in enumerate(PRO)
                 if np.isfinite(rho[i]) and float(rho[i]) >= UNR]
    _cross = [int(sv) for i, sv in enumerate(PRO) if np.isfinite(rho[i]) and float(rho[i]) >= 0.5]
    ell_reach_own = (min(_cross) if _cross else None)
    pert_rel_p = {str(s): float(np.mean([r['pert_rel'] for r in E4P[str(s)]])) for s in PRO}
    # com_layer 定义在**冻结 REACH** 上（同域配对），额外报 P16 新算域
    PRO_RE = [s for s in RE_L if s in PRO]
    xh_by = {int(s): float(xh[i]) for i, s in enumerate(PRO)}
    Jv_by = {int(s): float(Jv[i]) for i, s in enumerate(PRO)}
    rho_by = {int(s): float(rho[i]) for i, s in enumerate(PRO)}
    if len(PRO_RE) >= 2:
        com_layer_x = stat_com_layer(np.diff(np.array([xh_by[l] for l in PRO_RE], float)), PRO_RE)
        com_layer_j = stat_com_layer(np.diff(np.array([Jv_by[l] for l in PRO_RE], float)), PRO_RE)
        span3_x = stat_span_k(np.diff(np.array([xh_by[l] for l in PRO_RE], float)), 3)
        span3_j = stat_span_k(np.diff(np.array([Jv_by[l] for l in PRO_RE], float)), 3)
    else:
        com_layer_x = com_layer_j = span3_x = span3_j = None
    rec['E7_reach_own'] = dict(sites=PRO, reach=REACH_own, ell_reach=ell_reach_own,
                               unreach_y=UNR, full_swap=full_swap,
                               rho=[float(v) for v in rho])
    vp_h = [str(s) for s in A18['arms'][akey]['E7_summary']['sites_all']]
    w('  E7 [P] xhalf/J: %s'
      % '  '.join('L%d:%s' % (s, (F3(xh_by[s], 4) if np.isfinite(xh_by.get(s, np.nan)) else 'None'))
                  for s in PRO[:12]))
    w('  E7 [P] com_layer(x)=%s com_layer(J)=%s (域 n=%d) ; rho(α=1) 首值=%s'
      % (F3(com_layer_x), F3(com_layer_j), len(PRO_RE),
         F3(rho[0]) if len(rho) else 'NA'))

    # --- E8 零假设与确认集
    rg_c = np.random.default_rng(int(SEEDS['comB_inc']))
    rg_m = np.random.default_rng(int(SEEDS['comB_mlp']))
    rg_s = np.random.default_rng(int(SEEDS['share_mlp']))
    rg_x = np.random.default_rng(int(SEEDS['comlayer']))
    null_comB_inc = perm_null_com({l: abs(B['INC_ALL'][l]) for l in ALL_SITES}, RE_L, rg_c, BP)
    null_comB_mlp = perm_null_com({l: abs(B['INC_MLP'][l]) for l in ALL_SITES}, RE_L, rg_m, BP)
    null_share = perm_null_share(B['INC_MLP'], B['INC_ATTN'], RE_L, nb, rg_s, BP)
    null_comlayer_x = (perm_null_com({l: abs(xh_by[l]) for l in PRO_RE}, PRO_RE, rg_x, BP)
                       if len(PRO_RE) >= 2 else None)

    # --- E9 锚复现（仅 nf4 校准臂）
    an = ANCH.get(akey, {})
    ad = {}

    def chk(key, got, exp, tol=1e-6):
        ok = (got is not None and exp is not None and abs(got - exp) <= tol)
        ad[key] = dict(got=got, expected=exp, tol=tol, ok=bool(ok))
        return bool(ok)

    ok = True
    if scheme == 'nf4':
        ok &= chk('com_B_INC_ALL', com_B['INC_ALL'], an['com_B']['INC_ALL'], 1e-4)
        ok &= chk('com_B_INC_MLP', com_B['INC_MLP'], an['com_B']['INC_MLP'], 1e-4)
        ok &= chk('com_B_INC_ATTN', com_B['INC_ATTN'], an['com_B']['INC_ATTN'], 1e-4)
        ok &= chk('com_B_CUM_ALL', com_B['CUM_ALL'], an['com_B']['CUM_ALL'], 1e-4)
        ok &= chk('comlayer_B_all', comlayer_B_all, an['comlayer_B_all'], 1e-4)
        ok &= chk('comlayer_B_mlp', comlayer_B_mlp, an['comlayer_B_mlp'], 1e-4)
        ok &= chk('comlayer_B_attn', comlayer_B_attn, an['comlayer_B_attn'], 1e-4)
        ok &= chk('share_mlp_beh_nb', share_mlp_beh_nb, an['share_mlp_beh_nb'], 1e-4)
        ok &= chk('com_V_p17', com_V_recomputed, an['com_V'], 1e-4)
        ok &= chk('full_swap_p16', full_swap, an['full_swap'], 1e-4)
        ok &= chk('com_layer_x_p16', com_layer_x, an['com_layer_x'], 1e-4)
        ok &= chk('com_layer_j_p16', com_layer_j, an['com_layer_j'], 1e-4)
        ok &= chk('CUM_L_star', cum_bridge, an['cum_bridge'], 1e-4)
        ad['neighbourhood'] = dict(got=nb, expected=list(an['nb']), ok=bool(list(nb) == list(an['nb'])))
        ad['reach_len'] = dict(got=len(RE_L), expected=len(an['reach']),
                               ok=bool(len(RE_L) == len(an['reach'])))
        ok &= ad['neighbourhood']['ok'] and ad['reach_len']['ok']
    rec['E9_anchor'] = dict(ok=bool(ok), full_scale=bool(FULL_SCALE),
                            applies=bool(scheme == 'nf4' and FULL_SCALE), detail=ad,
                            note=('全尺度下适用' if FULL_SCALE else 'SMOKE/PROBE 缩幅网格 ⇒ 与冻结锚不可比，标注 N/A'))
    _app = rec['E9_anchor']['applies']
    w('  E9 锚复现[%s]: %s' % (scheme, ('OK' if ok else 'DRIFT') if _app
                              else 'N/A(%s)' % ('处理臂' if scheme != 'nf4' else '缩幅网格')))
    if _app and not ok:
        for k, v in ad.items():
            if not v['ok']:
                w('     !! %s got=%s expected=%s' % (k, v['got'], v['expected']))

    rec['E10_summary'] = dict(
        sites_all=ALL_SITES, reach=RE_L, nb=nb, L_star_own=L_STAR,
        com_B={c: com_B[c] for c in COMPONENTS},
        com_B_full={c: com_B_full[c] for c in COMPONENTS},
        com_B_conf={c: com_B_conf[c] for c in COMPONENTS_CONF},
        com_V_p17=float(an.get('com_V', float('nan'))) if an else None,
        com_V_recomputed=com_V_recomputed, com_V_own_spectrum=com_V_own,
        share_mlp_beh_nb=share_mlp_beh_nb, share_attn_beh_nb=share_attn_beh_nb,
        share_top1_beh_nb=share_top1_beh_nb, share_mlp_beh_reach=share_mlp_beh_reach,
        share_mlp_vec_nb=share_mlp_vec_nb, share_mlp_wown_nb=share_mlp_wown_nb,
        comlayer_B_all=comlayer_B_all, comlayer_B_mlp=comlayer_B_mlp, comlayer_B_attn=comlayer_B_attn,
        spearman_wall_ball=sp_wall_ball, spearman_wall_ball_own=sp_wall_ball_own,
        spearman_wmlp_bmlp_own=sp_wmlp_bmlp_own, spearman_wall_J=sp_wall_J,
        rlin_nb=rlin_nb, rlin_reach=rlin_reach, rlin_argmax=rlin_argmax,
        rlin_peak_ratio=rlin_peak_ratio,
        cum_bridge=cum_bridge, full_swap=full_swap, bridge_rel=bridge_rel,
        null_comB_inc=null_comB_inc, null_comB_mlp=null_comB_mlp, null_share=null_share,
        # 谱（跨口径配对用）
        b_all=[float(B['INC_ALL'][l]) for l in ALL_SITES],
        b_mlp=[float(B['INC_MLP'][l]) for l in ALL_SITES],
        b_attn=[float(B['INC_ATTN'][l]) for l in ALL_SITES],
        b_top1=[float(B['INC_TOP1'].get(l, float('nan'))) for l in ALL_SITES],
        b_cum=[float(B['CUM_ALL'].get(l, float('nan'))) for l in ALL_SITES],
        b_all_conf=[float(BC['INC_ALL'][l]) for l in ALL_SITES],
        w_own_all=[float(WV['INC_ALL'][l]) for l in ALL_SITES],
        w_own_mlp=[float(WV['INC_MLP'][l]) for l in ALL_SITES],
        w_own_attn=[float(WV['INC_ATTN'][l]) for l in ALL_SITES],
        pert_rel_inc=[float(SW_D['_rel']['INC_ALL'][l]) for l in ALL_SITES],
        pert_rel_cum=[float(SW_D['_rel']['CUM_ALL'][l]) for l in ALL_SITES],
        # Panel P
        profile_sites=PRO, alphas=ALPHAS, xh_frac=XHF,
        xhalf=[float(xh_by[s]) if np.isfinite(xh_by.get(s, np.nan)) else None for s in PRO],
        J=[float(Jv_by[s]) if np.isfinite(Jv_by.get(s, np.nan)) else None for s in PRO],
        rho=[float(rho_by[s]) for s in PRO],
        reach_own=REACH_own, ell_reach_own=ell_reach_own,
        com_layer_x=com_layer_x, com_layer_j=com_layer_j,
        com_layer_domain=PRO_RE, span3_x=span3_x, span3_j=span3_j,
        null_comlayer_x=null_comlayer_x,
        pert_rel_profile=pert_rel_p,
        xhalf_range=(float(np.nanmax(xh) - np.nanmin(xh)) if np.isfinite(xh).any() else None),
    )
    w('  E10 com_B(all)=%s com_B(mlp)=%s | share_mlp_beh(nb)=%s | comlayer_B_all=%s'
      % (F3(com_B['INC_ALL']), F3(com_B['INC_MLP']), F3(share_mlp_beh_nb, 4), F3(comlayer_B_all)))
    w('      com_V(P17 锚重算)=%s (本臂自谱 %s) ; spearman(w_all,b_all)[P17 w]=%s [本臂 w]=%s'
      % (F3(com_V_recomputed), F3(com_V_own), F3(sp_wall_ball, 4), F3(sp_wall_ball_own, 4)))
    w('      bridge CUM@L%d=%s vs FULL_SWAP=%s rel=%s'
      % (L_STAR, F3(cum_bridge), F3(full_swap), F3(bridge_rel, 4)))

    rec['n_forwards'] = int(n_fw)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rec


# ---------------------------------------------------------------- 判决
def per_arm_verdict(rec):
    E0 = rec['E0_selfcheck']
    FID = rec['E2_fidelity']
    S = rec['E10_summary']
    v = {}
    v['scheme'] = rec['scheme']
    v['Q0_device'] = rec['Q0_device']
    v['Q0_apparatus'] = bool(rec['T2_only'] and rec['F4_dims_ok'] and rec['F5_o_proj_ok']
                             and E0['determinism_maxdiff'] <= 1e-6 and E0['hook_effect_maxdiff'] > 1e-6)
    v['Q1_label'] = ('FID_PASS' if (FID['arch_max'] <= FL['P20_FID_ARCH']
                                    and FID['blk_max'] <= FL['P20_FID_BLK']) else 'FID_FAIL')
    v['Q1_arch_max'] = FID['arch_max']; v['Q1_blk_max'] = FID['blk_max']
    v['Q2_label'] = (('ANCHOR_OK' if rec['E9_anchor']['ok'] else 'ANCHOR_DRIFT')
                     if rec['E9_anchor']['applies'] else 'ANCHOR_NA_TREATMENT')
    v['Q2_detail'] = rec['E9_anchor']['detail']
    # 行为质心
    v['com_B_all'] = S['com_B']['INC_ALL']; v['com_B_mlp'] = S['com_B']['INC_MLP']
    v['com_B_attn'] = S['com_B']['INC_ATTN']; v['com_B_top1'] = S['com_B']['INC_TOP1']
    v['com_B_cum'] = S['com_B']['CUM_ALL']
    v['comlayer_B_all'] = S['comlayer_B_all']
    v['share_mlp_beh_nb'] = S['share_mlp_beh_nb']
    v['share_mlp_vec_nb'] = S['share_mlp_vec_nb']
    v['spearman_wall_ball'] = S['spearman_wall_ball']
    v['spearman_wall_ball_own'] = S['spearman_wall_ball_own']
    v['com_V'] = S['com_V_recomputed']
    v['median_reach'] = float(np.median(S['reach']))
    v['gap'] = (S['com_V_recomputed'] - S['com_B']['INC_ALL'])
    v['Q3_label'] = ('MLP_DOMINANT_BEH' if (S['share_mlp_beh_nb'] is not None
                                            and S['share_mlp_beh_nb'] > FL['MLP_DOM_MIN'])
                     else 'MLP_NOT_DOMINANT_BEH')
    v['Q4_label'] = ('SHALLOWER' if (v['gap'] >= FL['SHALLOWER_MIN'])
                     else ('DEEPER' if v['gap'] <= -FL['SHALLOWER_MIN'] else 'ALIGNED'))
    v['Q5_label'] = ('EFFICACY_COUPLED' if (S['spearman_wall_ball'] is not None
                                            and S['spearman_wall_ball'] > 0)
                     else 'EFFICACY_DECOUPLED')
    v['Q6_label'] = ('EFFICACY_COUPLED_OWN' if (S['spearman_wall_ball_own'] is not None
                                                and S['spearman_wall_ball_own'] > 0)
                     else 'EFFICACY_DECOUPLED_OWN')
    v['com_layer_x'] = S['com_layer_x']; v['com_layer_j'] = S['com_layer_j']
    v['rlin_nb'] = S['rlin_nb']; v['rlin_argmax'] = S['rlin_argmax']
    v['rlin_peak_ratio'] = S['rlin_peak_ratio']
    v['null_comB_inc'] = S['null_comB_inc']
    v['null_share'] = S['null_share']
    v['null_comlayer_x'] = S['null_comlayer_x']
    v['pert_rel_inc_max'] = float(np.nanmax(S['pert_rel_inc']))
    return v


def _arr(rec, key):
    return np.array([float(x) for x in rec['E10_summary'][key]], float)


def quant_pair_stats(V, recs, arm_n, arm_b):
    """同一个模型、两种口径的配对统计（唯一自变量 = 量化）。"""
    ra, rb = recs[arm_n], recs[arm_b]
    Sa, Sb = ra['E10_summary'], rb['E10_summary']
    sites = [l for l in Sa['sites_all'] if l in set(Sb['sites_all'])]
    RE_L = [l for l in Sa['reach'] if l in set(Sb['reach'])]
    va, vb = V[arm_n], V[arm_b]
    o = {}
    o['model'] = ra['model']
    o['arm_nf4'] = arm_n
    o['arm_bf16'] = arm_b
    o['n_sites'] = len(sites)
    o['n_reach'] = len(RE_L)
    o['com_V'] = dict(nf4=va['com_V'], bf16=vb['com_V'], delta=vb['com_V'] - va['com_V'])
    o['com_B_all'] = dict(nf4=va['com_B_all'], bf16=vb['com_B_all'], delta=vb['com_B_all'] - va['com_B_all'])
    o['com_B_mlp'] = dict(nf4=va['com_B_mlp'], bf16=vb['com_B_mlp'], delta=vb['com_B_mlp'] - va['com_B_mlp'])
    o['com_B_attn'] = dict(nf4=va['com_B_attn'], bf16=vb['com_B_attn'], delta=vb['com_B_attn'] - va['com_B_attn'])
    o['com_B_cum'] = dict(nf4=va['com_B_cum'], bf16=vb['com_B_cum'], delta=vb['com_B_cum'] - va['com_B_cum'])
    o['comlayer_B_all'] = dict(nf4=va['comlayer_B_all'], bf16=vb['comlayer_B_all'],
                               delta=(vb['comlayer_B_all'] - va['comlayer_B_all'])
                               if (va['comlayer_B_all'] is not None and vb['comlayer_B_all'] is not None) else None)
    o['com_layer_x'] = dict(nf4=va['com_layer_x'], bf16=vb['com_layer_x'],
                            delta=(vb['com_layer_x'] - va['com_layer_x'])
                            if (va['com_layer_x'] is not None and vb['com_layer_x'] is not None) else None)
    o['com_layer_j'] = dict(nf4=va['com_layer_j'], bf16=vb['com_layer_j'],
                            delta=(vb['com_layer_j'] - va['com_layer_j'])
                            if (va['com_layer_j'] is not None and vb['com_layer_j'] is not None) else None)
    o['share_mlp_beh_nb'] = dict(nf4=va['share_mlp_beh_nb'], bf16=vb['share_mlp_beh_nb'],
                                 delta=(vb['share_mlp_beh_nb'] - va['share_mlp_beh_nb'])
                                 if (va['share_mlp_beh_nb'] is not None and vb['share_mlp_beh_nb'] is not None) else None,
                                 same_side=bool((va['share_mlp_beh_nb'] > FL['MLP_DOM_MIN'])
                                                == (vb['share_mlp_beh_nb'] > FL['MLP_DOM_MIN'])))
    o['gap_sign_same'] = bool((va['gap'] >= FL['SHALLOWER_MIN']) == (vb['gap'] >= FL['SHALLOWER_MIN']))
    o['coupled_same'] = bool((va['Q5_label'] == 'EFFICACY_COUPLED') == (vb['Q5_label'] == 'EFFICACY_COUPLED'))
    # 谱一致性（同对象）
    def rho_and_resid(key_a, key_b):
        a = [Sa[key_a][l] for l in RE_L]; b = [Sb[key_b][l] for l in RE_L]
        r = spearman(a, b)
        aa = np.array(a, float); bb = np.array(b, float)
        den = np.maximum(np.abs(aa), 1e-9)
        res = np.abs(bb - aa) / den
        return (r, float(np.median(res)), float(np.percentile(res, 90)))
    idx = {l: i for i, l in enumerate(Sa['sites_all'])}
    idxb = {l: i for i, l in enumerate(Sb['sites_all'])}
    ia = [idx[l] for l in RE_L]; ib = [idxb[l] for l in RE_L]
    def rho_res_arr(ka, kb):
        a = [Sa[ka][i] for i in ia]; b = [Sb[kb][i] for i in ib]
        r = spearman(a, b)
        aa = np.array(a, float); bb = np.array(b, float)
        res = np.abs(bb - aa) / np.maximum(np.abs(aa), 1e-9)
        return r, float(np.median(res)), float(np.percentile(res, 90))
    for name, ka, kb in (('b_all', 'b_all', 'b_all'), ('b_mlp', 'b_mlp', 'b_mlp'),
                         ('b_attn', 'b_attn', 'b_attn'), ('w_own_all', 'w_own_all', 'w_own_all')):
        r, m, p90 = rho_res_arr(ka, kb)
        o['rho_' + name] = dict(rho=r, resid_med=m, resid_p90=p90)
    # xhalf 逐位点
    sxa = {s: Sa['xhalf'][i] for i, s in enumerate(Sa['profile_sites'])}
    sxb = {s: Sb['xhalf'][i] for i, s in enumerate(Sb['profile_sites'])}
    common = [s for s in sxa if s in sxb and sxa[s] is not None and sxb[s] is not None]
    dxh = [abs(sxa[s] - sxb[s]) for s in common]
    o['xhalf'] = dict(n_common=len(common), max_abs_dxh=(max(dxh) if dxh else None),
                      sites=common, nf4=[sxa[s] for s in common], bf16=[sxb[s] for s in common])
    sja = {s: Sa['J'][i] for i, s in enumerate(Sa['profile_sites'])}
    sjb = {s: Sb['J'][i] for i, s in enumerate(Sb['profile_sites'])}
    cj = [s for s in sja if s in sjb and sja[s] is not None and sjb[s] is not None
          and abs(sjb[s]) > 1e-9]
    jrat = [abs(sja[s] / sjb[s]) for s in cj]
    o['J'] = dict(n_common=len(cj), ratio_min=(min(jrat) if jrat else None),
                  ratio_max=(max(jrat) if jrat else None))
    o['reach_own_nf4'] = Sa['reach_own']; o['reach_own_bf16'] = Sb['reach_own']
    return o


def joint_verdict(V, recs, pairs):
    JV = {}
    arms = list(V.keys())
    JV['arms_present'] = arms
    JV['Q0_apparatus_all'] = all(V[a]['Q0_apparatus'] for a in arms)
    JV['Q0_device_all'] = all(V[a]['Q0_device'] in ('cuda', 'OFFLOAD') for a in arms)
    JV['Q1_joint'] = ('FID_ALL_PASS' if all(V[a]['Q1_label'] == 'FID_PASS' for a in arms)
                      else ('FID_PARTIAL' if any(V[a]['Q1_label'] == 'FID_PASS' for a in arms) else 'FID_FAIL'))
    cal = [a for a in arms if V[a]['scheme'] == 'nf4']
    JV['Q2_joint'] = ('ANCHOR_ALL_OK' if (cal and all(V[a]['Q2_label'] == 'ANCHOR_OK' for a in cal))
                      else 'ANCHOR_DRIFT')
    JV['Q2_calib_arms'] = cal
    # 量化稳定性
    def _d(p, k):
        d = p.get(k, {}).get('delta') if p.get(k) else None
        return abs(d) if d is not None else None
    JV['Q3_com_V_stable'] = all(_d(p, 'com_V') is not None and _d(p, 'com_V') <= FL['QUANT_TOL_COMV']
                                for p in pairs)
    JV['Q4_com_B_stable'] = all(_d(p, 'com_B_all') is not None and _d(p, 'com_B_all') <= FL['QUANT_TOL_COMB']
                                for p in pairs)
    JV['Q5_comlayer_stable'] = all(_d(p, 'comlayer_B_all') is not None
                                   and _d(p, 'comlayer_B_all') <= FL['QUANT_TOL_COMLAYER'] for p in pairs)
    JV['Q6_spectrum_consistent'] = all(p['rho_b_all']['rho'] is not None
                                       and p['rho_b_all']['rho'] >= FL['RHO_B_MIN'] for p in pairs)
    JV['Q7_share_stable'] = all((p['share_mlp_beh_nb']['same_side']
                                 and _d(p, 'share_mlp_beh_nb') is not None
                                 and _d(p, 'share_mlp_beh_nb') <= FL['QUANT_TOL_SHARE']) for p in pairs)
    JV['Q8_mlp_dom_retained'] = all(V[a]['Q3_label'] == 'MLP_DOMINANT_BEH' for a in arms)
    JV['Q9_shallow_retained'] = all(p['gap_sign_same'] for p in pairs)
    JV['Q10_coupled_retained'] = all(p['coupled_same'] for p in pairs)
    JV['Q11_profile_stable'] = all((_d(p, 'com_layer_x') is not None
                                    and _d(p, 'com_layer_x') <= FL['QUANT_TOL_COMLAYER']
                                    and _d(p, 'com_layer_j') is not None
                                    and _d(p, 'com_layer_j') <= FL['QUANT_TOL_COMLAYER']) for p in pairs)
    JV['Q12_xhalf_stable'] = all(p['xhalf']['max_abs_dxh'] is not None
                                 and p['xhalf']['max_abs_dxh'] <= FL['QUANT_TOL_XHALF'] for p in pairs)
    JV['quant_pairs'] = pairs
    return JV


def predictions_check(JV, V, recs, pairs):
    arms = JV['arms_present']
    P = {}
    P['P1'] = dict(name='装置与保真度（四臂）',
                   pass_=bool(JV['Q0_apparatus_all'] and JV['Q0_device_all']
                              and JV['Q1_joint'] == 'FID_ALL_PASS'),
                   detail=dict(Q0=JV['Q0_apparatus_all'], device=JV['Q0_device_all'], Q1=JV['Q1_joint'],
                               arch=[round(V[a]['Q1_arch_max'], 6) for a in arms],
                               blk=[round(V[a]['Q1_blk_max'], 6) for a in arms]))
    P['P2'] = dict(name='nf4 校准臂逐位复现 P18/P16 冻结锚',
                   pass_=bool(JV['Q2_joint'] == 'ANCHOR_ALL_OK'),
                   detail=dict(Q2=JV['Q2_joint'], calib=JV['Q2_calib_arms'],
                               anchors={a: recs[a]['E9_anchor']['detail'] for a in JV['Q2_calib_arms']}))
    _d = lambda p, k: (p.get(k) or {}).get('delta')
    P['P3'] = dict(name='holdout 主预测 1 —— 行为质心位置跨精度稳健（|Δcom_B(all)| ≤ 2.0 层，两对）',
                   pass_=bool(JV['Q4_com_B_stable']),
                   detail=dict(com_B={p['arm_nf4'] + '|' + p['arm_bf16']: _d(p, 'com_B_all') for p in pairs}))
    P['P4'] = dict(name='holdout 主预测 2 —— 行为谱形状跨精度一致（ρ(b_nf4,b_bf16) ≥ 0.80）',
                   pass_=bool(JV['Q6_spectrum_consistent']),
                   detail=dict(rho={p['arm_nf4'] + '|' + p['arm_bf16']: p['rho_b_all']['rho'] for p in pairs},
                               resid={p['arm_nf4']: dict(med=p['rho_b_all']['resid_med'],
                                                         p90=p['rho_b_all']['resid_p90']) for p in pairs}))
    P['P5'] = dict(name='holdout 主预测 3 —— 行为 MLP 主导在 bf16 下保持（同侧且 > 0.50）',
                   pass_=bool(JV['Q7_share_stable'] and JV['Q8_mlp_dom_retained']),
                   detail=dict(share={a: V[a]['share_mlp_beh_nb'] for a in arms},
                               same_side={p['arm_nf4']: p['share_mlp_beh_nb']['same_side'] for p in pairs},
                               Q8=JV['Q8_mlp_dom_retained']))
    P['P6'] = dict(name='holdout 主预测 4 —— 行为质心仍浅于向量质心（gap ≥ 2.0 层，两口径）',
                   pass_=bool(JV['Q9_shallow_retained']),
                   detail=dict(gap={a: V[a]['gap'] for a in arms},
                               same={p['arm_nf4']: p['gap_sign_same'] for p in pairs}))
    P['P7'] = dict(name='holdout 主预测 5 —— 同对象耦合（spearman(w,b) > 0）两口径皆成立',
                   pass_=bool(JV['Q10_coupled_retained']),
                   detail=dict(sp={a: V[a]['spearman_wall_ball'] for a in arms},
                               sp_own={a: V[a]['spearman_wall_ball_own'] for a in arms}))
    P['P8'] = dict(name='Panel P —— 写入窗剖面 com_layer(x)/com_layer(J) 跨精度稳健（≤2.0 层）',
                   pass_=bool(JV['Q11_profile_stable']),
                   detail=dict(com_layer_x={p['arm_nf4']: _d(p, 'com_layer_x') for p in pairs},
                               com_layer_j={p['arm_nf4']: _d(p, 'com_layer_j') for p in pairs}))
    P['P9'] = dict(name='Panel P —— 半饱和点 max|Δxhalf| ≤ 0.05（沿用 P16 的 XH_FAITHFUL_TOL）',
                   pass_=bool(JV['Q12_xhalf_stable']),
                   detail=dict(max_dxh={p['arm_nf4']: p['xhalf']['max_abs_dxh'] for p in pairs},
                               n={p['arm_nf4']: p['xhalf']['n_common'] for p in pairs}))
    P['P10'] = dict(name='对照（置换零假设 + 确认集 + 离流形诊断）', pass_=None,
                    detail=dict(null_comB={a: V[a]['null_comB_inc'].get('com_tail') for a in arms},
                                null_share={a: V[a]['null_share'].get('share_tail') for a in arms},
                                null_comlayer_x={a: (V[a]['null_comlayer_x'] or {}).get('com_tail')
                                                 for a in arms},
                                com_B_conf_delta={a: {c: (abs(recs[a]['E10_summary']['com_B_conf'][c]
                                                              - recs[a]['E10_summary']['com_B'][c])
                                                          if recs[a]['E10_summary']['com_B_conf'].get(c) is not None
                                                          else None)
                                                     for c in recs[a]['E10_summary']['com_B_conf']}
                                                 for a in arms},
                                pert_rel_inc_max={a: V[a]['pert_rel_inc_max'] for a in arms},
                                note='描述性条款。'))
    return P


def main():
    if MERGE:
        recs = {}
        for a in ARM_ORDER:
            p = os.path.join(P20T, ('_armrec20_probe_%s.json' % a) if PROBE else ('_armrec20_%s.json' % a))
            if os.path.exists(p):
                recs[a] = json.load(io.open(p, encoding='utf-8'))
        V = {a: per_arm_verdict(recs[a]) for a in recs}
        pairs = []
        for a in ARM_ORDER:
            if a.endswith('_nf4'):
                b = a[:-4] + '_bf16'
                if a in recs and b in recs:
                    pairs.append(quant_pair_stats(V, recs, a, b))
        JV = joint_verdict(V, recs, pairs)
        PRE = predictions_check(JV, V, recs, pairs)
        RES = dict(phase=20, line='N2h1-alpha-13', kind='result', smoke=SMOKE, probe=PROBE,
                   created_local=time.strftime('%Y-%m-%d %H:%M:%S'),
                   seal_sha256=EX['seal_sha256'], exec_sha256=sha(EXECP),
                   anchor_result_p18_sha256=EX['anchor_result_p18_sha256'],
                   anchor_result_p16_sha256=EX['anchor_result_p16_sha256'],
                   anchor_result_p17_sha256=EX['anchor_result_p17_sha256'],
                   arms=recs, verdict=V, joint_verdict=JV, predictions_check=PRE,
                   quant_pairs=pairs, floors=FL, bootstrap=EX['bootstrap'],
                   components=COMPONENTS, components_confirmation=COMPONENTS_CONF,
                   elapsed_total_s=(float(os.environ['ELAPSED_TOTAL'])
                                    if os.environ.get('ELAPSED_TOTAL') else None))
        sfx = '_probe' if PROBE else ('_smoke' if SMOKE else '')
        out = os.path.join(P20T, 'result_phase20%s.json' % sfx)
        io.open(out, 'w', encoding='utf-8').write(json.dumps(RES, ensure_ascii=False, indent=1))
        w('MERGED -> %s' % out)
        w('joint: %s' % json.dumps({k: JV[k] for k in JV
                                    if not k.startswith('quant_pairs') and k != 'Q2_detail'},
                                   ensure_ascii=False))
        w('predictions: %s' % json.dumps({k: PRE[k]['pass_'] for k in PRE}, ensure_ascii=False))
        for p in pairs:
            w('pair %s|%s: dcom_B=%s dcom_V=%s dcomlayer_B=%s rho_b=%s share %s->%s (same=%s) dxh=%s dCLx=%s dCLj=%s'
              % (p['arm_nf4'], p['arm_bf16'], F3(_d1(p, 'com_B_all')), F3(_d1(p, 'com_V')),
                 F3(_d1(p, 'comlayer_B_all')), F3(p['rho_b_all']['rho'], 4),
                 F3(p['share_mlp_beh_nb']['nf4'], 4), F3(p['share_mlp_beh_nb']['bf16'], 4),
                 p['share_mlp_beh_nb']['same_side'], F3(p['xhalf']['max_abs_dxh'], 4),
                 F3(_d1(p, 'com_layer_x')), F3(_d1(p, 'com_layer_j'))))
        return

    sel = [a for a in ARM_ORDER if (not ARMS_SEL) or any(a == s or a.startswith(s) for s in ARMS_SEL.split(','))]
    if PROBE and not ARMS_SEL:
        sel = [a for a in ARM_ORDER if a.startswith('A0')]
    if SMOKE and not ARMS_SEL:
        sel = [ARM_ORDER[0]]
    assert sel, 'ARMS 选择为空 (ARMS=%r)' % ARMS_SEL
    t_all = time.time()
    for arm_id in sel:
        w('')
        w('================ ARM %s ================' % arm_id)
        t0 = time.time()
        try:
            rec = run_arm(arm_id, ARMS_CFG[arm_id])
        except Exception:
            w('!! ARM %s FAILED' % arm_id)
            w(traceback.format_exc())
            raise
        rec['seconds'] = round(time.time() - t0, 1)
        dst = os.path.join(P20T, ('_armrec20_probe_%s.json' % arm_id) if PROBE
                           else ('_armrec20_%s.json' % arm_id))
        io.open(dst, 'w', encoding='utf-8').write(json.dumps(rec, ensure_ascii=False, indent=1))
        w('--- ARM %s DONE %.1fs -> %s' % (arm_id, rec['seconds'], dst))
    if SPLIT_PARTIAL:
        w('SPLIT_PARTIAL: 只写 partial，不合并')
    else:
        w('总耗时 %.1fs（未合并；合并请 MERGE=1）' % (time.time() - t_all))


def _d1(p, k):
    return (p.get(k) or {}).get('delta')


if __name__ == '__main__':
    main()
