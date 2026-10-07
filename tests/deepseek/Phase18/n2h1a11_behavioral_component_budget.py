# -*- coding: utf-8 -*-
"""
Phase 18 (N2h1-alpha-11) 主脚本：逐层组件「行为」预算。

Phase 17 把 Phase 8 的**向量**预算 share_v 从单层 L6 推广到逐层，得到向量质量谱 w_l 与质心 com_V，
并报告邻域 [26,28] 的**向量** MLP 份额 share_mlp_nb = 0.740 / 0.975 / 0.824（P5 MLP_DOMINANT_ALL）。
但那是几何量。P17 §7 的 H11 留下缺口：
  「若行为层同样 MLP 主导 => 升级为因果；若以 attn 为主 => P17 向量份额是几何假象」。

本 Phase 的**唯一**改动：把**同一批向量**从"量范数"改成"量行为"。
  注入物（完全沿用 P17 的定义，只是不再取范数）：
    INC_ALL : h_l^R + P_{U_l}(Delta_inc_l)      Delta_inc_l = Delta_attn_l + Delta_mlp_l
    INC_MLP : h_l^R + P_{U_l}(Delta_mlp_l)
    INC_ATTN: h_l^R + P_{U_l}(Delta_attn_l)
    INC_TOP1: h_l^R + P_{U_l}(Delta_head_h*_l)  h* = 该层 ||P_U(Delta_head)|| 最大的头
    CUM_ALL : h_l^R + P_{U_l}(d_l)              d_l = HH[l+1]^D - HH[l+1]^R（P16 的对象，桥接臂）
  读数（与 Phase 8 的 T 臂逐字同口径）：
    b_c,l = mean_pairs [ score_of(logits_patched, ds, sid_d) - BASE[rw].sd0 ]

**两条不可混用的口径**（写进 seal）：
  (1) 向量预算的"精确可加"是**定义**（Delta_inc := Delta_attn + Delta_mlp）；**行为量不可加**，
      故另设**线性残差** r_lin,l = |b_all - (b_mlp + b_attn)| / max(|b_all|, eps) —— 它是诊断，不是误差。
  (2) 份额判据用**同结构比**：share_mlp_beh = sum_{l in nb} |b_mlp,l| / sum_{l in nb} |b_all,l|，
      与 P17 的 share_mlp_nb = sum w^mlp_l / sum w_l 结构逐项对应。

用法：SMOKE=1 python n2h1a11_behavioral_component_budget.py
      SPLIT_PARTIAL=1 ARMS=A0 python ...        (逐臂进程隔离)
      MERGE=1 python ...                        (合并三臂 -> result)
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
P18 = os.path.join(ROOT, 'tests', 'deepseek', 'Phase18')
P18T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase18')
P17T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase17')
P16T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase16')
EXECP = os.path.join(P18T, 'execution_phase18.json')
SEALP = os.path.join(P18T, 'N2h1a11_design_seal.json')
SMOKE = os.environ.get('SMOKE', '0') == '1'
ARMS_SEL = os.environ.get('ARMS', '')
SPLIT_PARTIAL = os.environ.get('SPLIT_PARTIAL', '0') == '1'
MERGE = os.environ.get('MERGE', '0') == '1'
os.makedirs(P18T, exist_ok=True)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def sha8(p):
    return sha(p)[:8]


EX = json.load(io.open(EXECP, encoding='utf-8'))
assert sha(SEALP) == EX['seal_sha256'], 'DRIFT: seal sha != execution.seal_sha256'

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
QUANT = EX['quant']
COMPONENTS = list(EX['components'])
COMPONENTS_CONF = list(EX['components_confirmation'])
FL = EX['floors']
BP = int(EX['bootstrap']['BP'])
SEEDS = dict(EX['bootstrap']['seeds'])
ARM_ORDER = list(EX['arm_order'])
ARMS_CFG = EX['arms']
NBW = int(EX['neighbourhood_width'])

if SMOKE:
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

_log = []


def w(s=''):
    _log.append(str(s))
    print(s)
    sys.stdout.flush()


def F3(v, nd=3):
    return ('%.' + str(nd) + 'f') % v if isinstance(v, (int, float)) and v is not None else str(v)


# ---------------------------------------------------------------- 统计工具
def stat_com_layer(jumps, sites):
    """与 Phase 16/17 逐字节同口径：mid = 相邻位点中点，com = sum|j|*mid / sum|j|。"""
    j = np.asarray(jumps, float)
    if len(j) == 0 or len(sites) != len(j) + 1:
        return None
    mid = (np.asarray(sites, float)[:-1] + np.asarray(sites, float)[1:]) / 2.0
    a = np.abs(j)
    den = float(a.sum())
    if not np.isfinite(den) or den <= 1e-12:
        return None
    return float((a * mid).sum() / den)


def com_of_mass(mass_by_site, sites):
    """区间求和质心（与 P17 的 com_of_mass 逐字节同口径）。
    W_j = sum_{l in [s_j, s_{j+1})} mass[l]；mid_j = (s_j+s_{j+1})/2；com = sum W_j mid_j / sum W_j。"""
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
    """质心的置换零假设：保留质量多重集，随机重排到位点（顺序敏感 => 非退化）。"""
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
        return dict(BP=n_bp, n_ok=0, obs_com=obs, com_p5=None, com_p95=None, com_tail=None, reason='all_nan')
    p5, p95 = float(np.percentile(fin, 5)), float(np.percentile(fin, 95))
    tail = ('low' if (obs is not None and obs <= p5) else
            'high' if (obs is not None and obs >= p95) else 'none')
    return dict(BP=n_bp, n_ok=int(len(fin)), obs_com=obs, com_p5=p5, com_p95=p95,
                com_tail=tail, reason=None)


def perm_null_share(b_mlp, b_attn, sites_reach, nb, rng_obj, n_bp):
    """份额的置换零假设：固定 nb（位置子集），只把 **MLP 质量** 随机重排到 REACH 位点上。
    注意：若在**全集**上取份额，置换不变 => 结构性退化。故必须锚在固定子集 nb 上。"""
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


# ---------------------------------------------------------------- 单臂
def run_arm(arm_id, acfg):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    rec = dict(arm=arm_id, role=acfg['role'], model=acfg['model'], smoke=SMOKE)
    MDIR = os.path.join(ROOT, 'models', 'hf', acfg['dir'])

    def ids_of(s):
        return tok.encode(s, add_special_tokens=False)

    def attn_out_proj(layer):
        a = layer.self_attn
        for nm in ('o_proj', 'dense', 'out_proj'):
            if hasattr(a, nm):
                return getattr(a, nm), nm
        raise RuntimeError('no attn out proj')

    tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)

    # --- F1b 类别 token 逐臂现场解析
    SUP_ID = {}
    for _wd in SUPS:
        _t = list(tok.encode(_wd, add_special_tokens=False))
        assert len(_t) == 1 and tok.decode([_t[0]]) == _wd, \
            'F1b 失败：类别词 %r 非单 token 或 decode 不可逆 %r' % (_wd, _t)
        SUP_ID[_wd] = int(_t[0])
    rec['sup_id_arm'] = dict(SUP_ID)
    rec['F1b_ok'] = True

    # --- F1 T=2 布局
    tl = {}
    for wd, sup in INST_ALL:
        tl.setdefault(len(ids_of(TMPL % wd)), []).append(wd)
    rec['token_len_hist'] = {str(k): len(v) for k, v in sorted(tl.items())}
    rec['T2_only'] = bool(sorted(tl.keys()) == [2])
    w('  F1 T=2: hist=%s only=%s | F1b sup_id=%s' % (rec['token_len_hist'], rec['T2_only'],
                                                    json.dumps(SUP_ID, ensure_ascii=False)))

    max_mem = {int(k) if str(k).isdigit() else k: v for k, v in QUANT['max_memory'].items()}
    bnb = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_quant_type=QUANT['bnb_4bit_quant_type'],
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=bool(QUANT['bnb_4bit_use_double_quant']))
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MDIR, quantization_config=bnb, trust_remote_code=True,
        attn_implementation=QUANT['attn_implementation'], low_cpu_mem_usage=True,
        device_map=QUANT['device_map'], max_memory=max_mem)
    model.eval()
    rec['load_s'] = round(time.time() - t0, 1)
    w('  loaded %.1fs' % rec['load_s'])

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
    rec['config_sha256'] = sha(os.path.join(MDIR, 'config.json'))
    dev_hist = {}
    for nm_, p in model.named_parameters():
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
    Q0_device = 'cuda' if all(k.startswith('cuda') for k in dev_hist) else 'MIXED'
    rec['Q0_device'] = Q0_device

    # 位点 = 层号 l（注入 hooks layers[l] 的输出 = HH[l+1]）；与 P17 的 w_all（layer 0..L-2）逐索引对齐。
    ALL_SITES = list(range(1, L - 1))
    REACH = [int(x) for x in EX['p16_anchors'][arm_id]['reach']]
    RE_L = [s for s in REACH if s in ALL_SITES]
    nb = [s for s in RE_L if abs(s - EX['p17_anchors'][arm_id]['com_V']) <= NBW]
    BRIDGE_SITE = int(EX['p16_anchors'][arm_id]['L_star_own'])   # 该臂自己的写入窗
    w('  全域位点 n=%d ; REACH n=%d ; P17 邻域 nb=%s' % (len(ALL_SITES), len(RE_L), nb))
    assert nb, 'nb 为空 —— P17 com_V 或 REACH 锚有问题'

    def ids_t(text):
        return torch.tensor([ids_of(text)], device='cuda')

    # --- E0 装置自检
    n_fw = 0
    with torch.no_grad():
        lga = model(input_ids=ids_t(TMPL % '苹果')).logits[0, -1].float().detach().cpu().numpy()
        lgb = model(input_ids=ids_t(TMPL % '苹果')).logits[0, -1].float().detach().cpu().numpy()
        n_fw += 2
    determinism = float(np.max(np.abs(lga - lgb)))
    rng0 = np.random.default_rng(EX['bootstrap']['seed'])
    vec0 = (rng0.standard_normal(HID) * 0.1).astype(np.float32)
    site0 = L // 2

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

    lg_hook = fwd_patch(TMPL % '苹果', site0, torch.tensor(vec0, device='cuda'))
    n_fw += 1
    hook_effect = float(np.max(np.abs(lg_hook - lgb)))
    rec['E0_selfcheck'] = dict(determinism_maxdiff=determinism, hook_site=site0,
                              hook_effect_maxdiff=hook_effect, Q0_device=Q0_device,
                              F4_dims_ok=rec['F4_dims_ok'], F5_o_proj_ok=rec['F5_o_proj_ok'])
    w('  E0 自检: determinism=%.3e ; hook@L%d effect=%.3e ; device=%s'
      % (determinism, site0, hook_effect, Q0_device))

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

    # --- 逐头分块（模块自身前向）
    NH1 = NH + 1
    blocks = np.zeros((NH1, OIN), np.float32)

    @torch.no_grad()
    def head_blocks(opmod, v):
        blocks[:] = 0.0
        for h in range(NH):
            blocks[h, h * HDP:(h + 1) * HDP] = v[h * HDP:(h + 1) * HDP]
        blocks[NH] = v
        t = torch.tensor(blocks, device='cuda', dtype=torch.bfloat16)
        o = opmod(t).float().cpu().numpy()
        return o[:NH], o[NH]

    # --- E2 保真度门
    arch, blk = [], []
    for wd, sup in INST_ALL[:6]:
        HHs, O, M, _ = CAP[wd]
        for l in range(L - 1):
            lhs = HHs[l + 1] - HHs[l]
            rhs = OPR[l](torch.tensor(O[l][None, :], device='cuda', dtype=torch.bfloat16)
                         ).float().cpu().numpy()[0] + M[l]
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

    # --- U_l（与 P16/P17 同口径）
    by = {}
    for wd, sup in INST_ALL:
        by.setdefault(sup, []).append(wd)
    AVAIL = [s for s in SUPS if s in by]
    rk = max(len(AVAIL) - 1, 1)
    U = {}
    for l in range(L):
        mus = np.stack([np.mean([CAP[wd][0][l + 1] for wd in by[s]], 0) for s in AVAIL], 0).astype(np.float64)
        Dm = mus - mus.mean(0, keepdims=True)
        _, sv, Vt = np.linalg.svd(Dm, full_matrices=False)
        U[l] = Vt[:rk].astype(np.float32)
    rec['E3_U'] = dict(rank=rk, n_classes=len(AVAIL), classes=AVAIL)
    w('  E3 U_ell: rank=%d n_classes=%d' % (rk, len(AVAIL)))

    def proj(v, Ub):
        return (v @ Ub.T) @ Ub

    # --- E4 BASE + FULL_SWAP（从 P16 result 现场读，同 discovery 对）
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
    w('  E4 base: n=%d ; bad=%s ; disc=%d conf=%d'
      % (len(BASE), bad_base if bad_base else 'NONE', len(disc_pairs), len(conf_pairs)))
    rec['E4_base'] = dict(n=len(BASE), bad=bad_base, n_disc=len(disc_pairs), n_conf=len(conf_pairs))
    FS_PAIR = A16['arms'][arm_id]['E2_full_swap']['FS_PAIR']
    full_swap = float(np.mean([float(FS_PAIR[p[0]]) for p in disc_pairs if p[0] in FS_PAIR]))
    rec['E4_full_swap'] = dict(FULL_SWAP=full_swap, n=len(disc_pairs))

    # --- E5 组件注入扫描
    def sweep(pairs, comps, sites):
        out = {}
        for cname in comps:
            b = {}
            rel = {}
            per = {}
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
                        pv = np.array([float(np.linalg.norm(proj(dd_h[h], Ub))) for h in range(NH)])
                        dv = dd_h[int(np.argmax(pv))]
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
                n = max(len(pairs), 1)
                b[s] = dd / n
                rel[s] = float(np.mean(rl)); per[s] = pp
                out.setdefault('_rel', {})[cname] = rel
                out.setdefault('_pvnorm', {})[cname] = pr
            out[cname] = b
            out.setdefault('_per_pair', {})[cname] = per
        return out

    t0 = time.time()
    SW_D = sweep(disc_pairs, COMPONENTS, ALL_SITES)
    n_fw += len(disc_pairs) * len(COMPONENTS) * len(ALL_SITES)
    rec['E5_seconds'] = round(time.time() - t0, 1)
    w('  E5 discovery 扫描 %d 组件 x %d 位点 x %d 对 / %.1fs'
      % (len(COMPONENTS), len(ALL_SITES), len(disc_pairs), rec['E5_seconds']))

    t0 = time.time()
    SW_C = sweep(conf_pairs, COMPONENTS_CONF, ALL_SITES)
    n_fw += len(conf_pairs) * len(COMPONENTS_CONF) * len(ALL_SITES)
    rec['E6_seconds'] = round(time.time() - t0, 1)
    w('  E6 confirmation 扫描 %d 组件 x %d 位点 x %d 对 / %.1fs'
      % (len(COMPONENTS_CONF), len(ALL_SITES), len(conf_pairs), rec['E6_seconds']))

    # --- E7 汇总
    def bmap(sw, c):
        return {int(l): float(sw[c][l]) for l in ALL_SITES}

    B = {c: bmap(SW_D, c) for c in COMPONENTS}
    BC = {c: bmap(SW_C, c) for c in COMPONENTS_CONF}
    # P17 的 w 谱（现场读入，用于 spearman 与口径对照）
    W = {k: {int(l): float(v) for l, v in enumerate(A17['arms'][arm_id]['E5_com_V'][k])}
         for k in ('w_all', 'w_mlp', 'w_attn')}

    # 质心：abs(b) 的区间和质心（全域 & REACH 两个域都报）
    def cen(c, sites, table=None):
        return com_of_mass({l: abs((table or B)[c][l]) for l in ALL_SITES}, sites)[0]

    com_B = {c: cen(c, RE_L) for c in COMPONENTS}
    com_B_full = {c: cen(c, ALL_SITES) for c in COMPONENTS}
    com_B_conf = {c: cen(c, RE_L, BC) for c in COMPONENTS_CONF}
    # P17 的 com_V 同域重算（口径核对：应等于 P17 记录的 com_V）
    com_V_recomputed = com_of_mass({l: W['w_all'][l] for l in ALL_SITES}, RE_L)[0]

    # 邻域份额（行为）
    def sh(c_num, c_den, sites, table=None):
        t = table or B
        num = float(sum(abs(t[c_num][l]) for l in sites))
        den = float(sum(abs(t[c_den][l]) for l in sites))
        return (num / den) if den > 1e-12 else None, num, den

    share_mlp_beh_nb, s_m, s_a_ = sh('INC_MLP', 'INC_ALL', nb)
    share_attn_beh_nb, s_a, _ = sh('INC_ATTN', 'INC_ALL', nb)
    share_mlp_beh_reach, _, _ = sh('INC_MLP', 'INC_ALL', RE_L)
    share_attn_beh_reach, _, _ = sh('INC_ATTN', 'INC_ALL', RE_L)
    share_top1_beh_nb, s_t1, _ = sh('INC_TOP1', 'INC_ALL', nb)
    # 对照：P17 的**向量**份额（同 nb 口径，现场从 w 谱重算）
    v_m = float(sum(W['w_mlp'][l] for l in nb)); v_a = float(sum(W['w_attn'][l] for l in nb))
    v_all = float(sum(W['w_all'][l] for l in nb))
    share_mlp_vec_nb = (v_m / v_all) if v_all > 1e-12 else None
    share_attn_vec_nb = (v_a / v_all) if v_all > 1e-12 else None

    # 行为剖面的 com_layer（对 b 取相邻差，与 P16/P17 的 com_layer 同口径）
    def com_layer_of(c, table=None):
        t = table or B
        arr = np.array([t[c][l] for l in RE_L], float)
        return stat_com_layer(np.diff(arr), RE_L)

    comlayer_B_all = com_layer_of('INC_ALL')
    comlayer_B_mlp = com_layer_of('INC_MLP')
    comlayer_B_attn = com_layer_of('INC_ATTN')

    # spearman（同对象对齐：w 与 b 在 REACH 位点上）
    def sper(kw, kb, table=None):
        t = table or B
        return spearman([W[kw][l] for l in RE_L], [abs(t[kb][l]) for l in RE_L])

    sp_wall_ball = sper('w_all', 'INC_ALL')
    sp_wmlp_bmlp = sper('w_mlp', 'INC_MLP')
    sp_wattn_battn = sper('w_attn', 'INC_ATTN')
    # 对照：P17 的 P6 口径（w vs P16 的 J）
    J_p16 = {int(s): float(v) for s, v in zip(A16['E4_summary'][arm_id]['sites'],
                                              A16['E4_summary'][arm_id]['J'])}
    xl = [l for l in RE_L if l in J_p16]
    sp_wall_J = spearman([W['w_all'][l] for l in xl], [J_p16[l] for l in xl])

    # 线性残差
    rlin = {int(l): (abs(B['INC_ALL'][l] - (B['INC_MLP'][l] + B['INC_ATTN'][l])) /
                     max(abs(B['INC_ALL'][l]), 1e-12)) for l in ALL_SITES}
    rlin_nb = float(np.mean([rlin[l] for l in nb]))
    rlin_reach = float(np.mean([rlin[l] for l in RE_L]))
    _rl = np.array([rlin[l] for l in ALL_SITES], float)
    _order = np.argsort(-_rl)
    rlin_argmax = int(ALL_SITES[int(_order[0])])
    rlin_peak = float(_rl[_order[0]])
    rlin_second = float(_rl[_order[1]]) if len(_order) > 1 else None
    rlin_peak_ratio = (rlin_peak / rlin_second) if (rlin_second and rlin_second > 1e-12) else None
    rlin_at_lstar = rlin.get(int(EX['p16_anchors'][arm_id]['L_star_own']))
    # 份额的第二种分母（(mlp+attn) 为分母，与置换零假设同口径）
    _nm = float(sum(abs(B['INC_MLP'][l]) for l in nb)); _na = float(sum(abs(B['INC_ATTN'][l]) for l in nb))
    share_ratio_mlp_attn_nb = (_nm / (_nm + _na)) if (_nm + _na) > 1e-12 else None
    _wv_re = float(sum(W['w_mlp'][l] for l in RE_L)); _wv_ra = float(sum(W['w_all'][l] for l in RE_L))
    share_mlp_vec_reach = (_wv_re / _wv_ra) if _wv_ra > 1e-12 else None

    # 桥接：CUM@BRIDGE_SITE vs P16 FULL_SWAP
    cum_bridge = float(B['CUM_ALL'].get(BRIDGE_SITE, float('nan')))
    bridge_rel = (abs(cum_bridge - full_swap) / max(abs(full_swap), 1e-12)
                  if np.isfinite(cum_bridge) else None)

    # 零假设
    rg_c = np.random.default_rng(int(SEEDS['comB_inc']))
    rg_m = np.random.default_rng(int(SEEDS['comB_mlp']))
    rg_s = np.random.default_rng(int(SEEDS['share_mlp']))
    null_comB_inc = perm_null_com({l: abs(B['INC_ALL'][l]) for l in ALL_SITES}, RE_L, rg_c, BP)
    null_comB_mlp = perm_null_com({l: abs(B['INC_MLP'][l]) for l in ALL_SITES}, RE_L, rg_m, BP)
    null_share = perm_null_share(B['INC_MLP'], B['INC_ATTN'], RE_L, nb, rg_s, BP)

    rec['E7_summary'] = dict(
        sites_all=ALL_SITES, reach=RE_L, nb=nb,
        com_B=com_B, com_B_full=com_B_full, com_B_conf=com_B_conf,
        com_V_p17=float(EX['p17_anchors'][arm_id]['com_V']),
        com_V_recomputed=com_V_recomputed,
        share_mlp_beh_nb=share_mlp_beh_nb, share_attn_beh_nb=share_attn_beh_nb,
        share_top1_beh_nb=share_top1_beh_nb,
        share_mlp_beh_reach=share_mlp_beh_reach, share_attn_beh_reach=share_attn_beh_reach,
        share_mlp_vec_nb=share_mlp_vec_nb, share_attn_vec_nb=share_attn_vec_nb,
        comlayer_B_all=comlayer_B_all, comlayer_B_mlp=comlayer_B_mlp, comlayer_B_attn=comlayer_B_attn,
        spearman_wall_ball=sp_wall_ball, spearman_wmlp_bmlp=sp_wmlp_bmlp,
        spearman_wattn_battn=sp_wattn_battn, spearman_wall_J=sp_wall_J,
        rlin_nb=rlin_nb, rlin_reach=rlin_reach, rlin_at_bridge=rlin.get(BRIDGE_SITE),
        rlin_argmax=rlin_argmax, rlin_peak=rlin_peak, rlin_peak_ratio=rlin_peak_ratio,
        rlin_at_lstar=rlin_at_lstar, L_star_own=int(EX['p16_anchors'][arm_id]['L_star_own']),
        share_ratio_mlp_attn_nb=share_ratio_mlp_attn_nb, share_mlp_vec_reach=share_mlp_vec_reach,
        bridge_site=BRIDGE_SITE,
        rlin_by_site={str(l): rlin[l] for l in ALL_SITES},
        cum_bridge=cum_bridge, full_swap=full_swap, bridge_rel=bridge_rel,
        null_comB_inc=null_comB_inc, null_comB_mlp=null_comB_mlp, null_share=null_share,
        b_all=[float(B['INC_ALL'].get(l, float('nan'))) for l in ALL_SITES],
        b_mlp=[float(B['INC_MLP'].get(l, float('nan'))) for l in ALL_SITES],
        b_attn=[float(B['INC_ATTN'].get(l, float('nan'))) for l in ALL_SITES],
        b_top1=[float(B['INC_TOP1'].get(l, float('nan'))) for l in ALL_SITES],
        b_cum=[float(B['CUM_ALL'].get(l, float('nan'))) for l in ALL_SITES],
        b_all_conf=[float(BC['INC_ALL'].get(l, float('nan'))) for l in ALL_SITES],
        pert_rel_inc=[float(SW_D['_rel']['INC_ALL'][l]) for l in ALL_SITES],
        pert_rel_cum=[float(SW_D['_rel']['CUM_ALL'][l]) for l in ALL_SITES],
        per_pair_inc={str(l): SW_D['_per_pair']['INC_ALL'][l] for l in ALL_SITES},
        per_pair_mlp={str(l): SW_D['_per_pair']['INC_MLP'][l] for l in ALL_SITES},
        per_pair_attn={str(l): SW_D['_per_pair']['INC_ATTN'][l] for l in ALL_SITES},
    )
    w('  E7 com_B(inc)=%s (全域 %s) ; com_V(P17)=%s (同域重算 %s, drift %.3e)'
      % (F3(com_B['INC_ALL']), F3(com_B_full['INC_ALL']),
         F3(float(EX['p17_anchors'][arm_id]['com_V'])),
         F3(com_V_recomputed), abs(float(com_V_recomputed) - float(EX['p17_anchors'][arm_id]['com_V']))))
    w('     邻域 nb=%s : share_mlp_beh=%s (ratio %s ; 向量 %s) ; share_top1_beh=%s ; rlin_nb=%s'
      % (nb, F3(share_mlp_beh_nb, 4), F3(share_ratio_mlp_attn_nb, 4), F3(share_mlp_vec_nb, 4),
         F3(share_top1_beh_nb, 4), F3(rlin_nb, 4)))
    w('     spearman(w_all,b_all)=%s ; (w_mlp,b_mlp)=%s ; (w_attn,b_attn)=%s ; [P17 口径 (w,J)]=%s'
      % (F3(sp_wall_ball, 4), F3(sp_wmlp_bmlp, 4), F3(sp_wattn_battn, 4), F3(sp_wall_J, 4)))
    w('     bridge: CUM@L%d=%s vs FULL_SWAP=%s rel=%s ; rlin@L*=%s ; argmax rlin=%s (比值 %s)'
      % (BRIDGE_SITE, F3(cum_bridge), F3(full_swap), F3(bridge_rel, 4), F3(rlin_at_lstar, 4),
         rlin_argmax, F3(rlin_peak_ratio, 3)))

    # --- E8 锚复现（P16 五条 + P17 三条）
    an16 = EX['p16_anchors'][arm_id]
    an17 = EX['p17_anchors'][arm_id]
    ad = {}

    def chk(key, got, exp, tol=1e-6):
        ok = (got is not None and exp is not None and abs(got - exp) <= tol)
        ad[key] = dict(got=got, expected=exp, ok=bool(ok))
        return bool(ok)

    xh = A16['E4_summary'][arm_id]['xhalf']
    sites16 = [int(s) for s in A16['E4_summary'][arm_id]['sites']]
    xh_by = {int(s): float(v) for s, v in zip(sites16, xh)}
    jumps_x = np.diff(np.array([xh_by[l] for l in RE_L], float))
    jumps_j = np.diff(np.array([J_p16[l] for l in RE_L], float))
    ok = True
    ok &= chk('com_layer_x', stat_com_layer(jumps_x, RE_L), an16['com_layer_x'])
    ok &= chk('com_layer_j', stat_com_layer(jumps_j, RE_L), an16['com_layer_j'])
    ok &= chk('L_star_own', A16['E3_localize'][arm_id]['L_star_own'], an16['L_star_own'], 0)
    ok &= chk('ell_reach', A16['E7_reach'][arm_id]['ell_reach'], an16['ell_reach'], 0)
    ok &= chk('reach_len', float(len(RE_L)), float(len(an16['reach'])), 0)
    ok &= chk('com_V_p17', com_V_recomputed, an17['com_V'], 1e-4)
    ok &= chk('share_mlp_vec_nb', share_mlp_vec_nb, an17['share_mlp_nb'], 1e-4)
    ad['neighbourhood'] = dict(got=nb, expected=list(an17['neighbourhood']),
                               ok=bool(list(nb) == list(an17['neighbourhood'])))
    ok &= ad['neighbourhood']['ok']
    rec['E8_anchor'] = dict(ok=bool(ok), detail=ad)
    w('  E8 锚复现: %s' % ('OK' if ok else 'DRIFT'))
    if not ok:
        for k, v in ad.items():
            if not v['ok']:
                w('     !! %s got=%s expected=%s' % (k, v['got'], v['expected']))

    rec['n_forwards'] = int(n_fw)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rec


# ---------------------------------------------------------------- 判决
def per_arm_verdict(rec):
    E0 = rec['E0_selfcheck']
    FID = rec['E2_fidelity']
    S = rec['E7_summary']
    v = {}
    v['Q0_device'] = rec['Q0_device']
    v['Q0_apparatus'] = bool(rec['T2_only'] and rec['F4_dims_ok'] and rec['F5_o_proj_ok']
                             and E0['determinism_maxdiff'] <= 1e-6 and E0['hook_effect_maxdiff'] > 1e-6)
    v['Q1_label'] = ('FID_PASS' if (FID['arch_max'] <= FL['P18_FID_ARCH']
                                    and FID['blk_max'] <= FL['P18_FID_BLK']) else 'FID_FAIL')
    v['Q1_arch_max'] = FID['arch_max']; v['Q1_blk_max'] = FID['blk_max']
    v['Q2_label'] = ('ANCHOR_OK' if rec['E8_anchor']['ok'] else 'ANCHOR_DRIFT')
    v['Q2_detail'] = rec['E8_anchor']['detail']
    # Q3 桥接
    br = S['bridge_rel']
    v['Q3_bridge_rel'] = br; v['Q3_cum_bridge'] = S['cum_bridge']; v['Q3_full_swap'] = S['full_swap']
    v['Q3_label'] = ('BRIDGE_OK' if (br is not None and br <= FL['BRIDGE_TOL_CUM']) else 'BRIDGE_DRIFT')
    # Q4 行为组件归属
    sm = S['share_mlp_beh_nb']
    v['Q4_share_mlp_beh_nb'] = sm
    v['Q4_share_mlp_vec_nb'] = S['share_mlp_vec_nb']
    v['Q4_share_top1_beh_nb'] = S['share_top1_beh_nb']
    v['Q4_rlin_nb'] = S['rlin_nb']
    v['Q4_share_ratio_mlp_attn_nb'] = S['share_ratio_mlp_attn_nb']
    v['Q4_share_mlp_vec_reach'] = S['share_mlp_vec_reach']
    v['Q4_label'] = ('MLP_DOMINANT_BEH' if (sm is not None and sm > FL['MLP_DOM_MIN'])
                     else 'MLP_NOT_DOMINANT_BEH')
    # Q5 向量 vs 行为归属一致性（H11）
    sv = S['share_mlp_vec_nb']
    if (sm is None or sv is None):
        v['Q5_label'] = 'ATTRIBUTION_UNDEFINED'
    else:
        same = ((sm > FL['MLP_DOM_MIN']) == (sv > FL['MLP_DOM_MIN']))
        v['Q5_label'] = 'ATTRIBUTION_CONSISTENT' if same else 'ATTRIBUTION_DISCREPANT'
    # Q6 深度：com_B vs com_V 与 median(REACH)
    cb = S['com_B']['INC_ALL']
    v['Q6_com_B'] = cb; v['Q6_com_V'] = S['com_V_p17']
    v['Q6_median_reach'] = float(np.median(S['reach']))
    v['Q6_gap'] = (S['com_V_p17'] - cb) if cb is not None else None
    v['Q6_label'] = ('SHALLOWER' if (cb is not None and v['Q6_gap'] >= FL['SHALLOWER_MIN'])
                     else ('DEEPER' if (cb is not None and v['Q6_gap'] <= -FL['SHALLOWER_MIN'])
                           else 'ALIGNED'))
    v['Q6_deep_label'] = ('DEEP' if (cb is not None and cb >= v['Q6_median_reach']) else 'SHALLOW')
    # Q7 同对象耦合
    sp = S['spearman_wall_ball']
    v['Q7_spearman_wall_ball'] = sp
    v['Q7_spearman_wall_J'] = S['spearman_wall_J']
    v['Q7_spearman_wmlp_bmlp'] = S['spearman_wmlp_bmlp']
    v['Q7_label'] = ('EFFICACY_COUPLED' if (sp is not None and sp > 0) else 'EFFICACY_DECOUPLED')
    # Q8 超可加性（写窗 vs 深端）
    v['Q8_rlin_bridge'] = S['rlin_at_bridge']; v['Q8_rlin_nb'] = S['rlin_nb']
    v['Q8_rlin_reach'] = S['rlin_reach']
    v['Q8_rlin_argmax'] = S['rlin_argmax']; v['Q8_L_star_own'] = S['L_star_own']
    v['Q8_rlin_peak_ratio'] = S['rlin_peak_ratio']
    v['Q8_label'] = ('SUPERADD_AT_WINDOW'
                     if (S['rlin_argmax'] == S['L_star_own']
                         and S['rlin_peak_ratio'] is not None
                         and S['rlin_peak_ratio'] >= FL['RLIN_PEAK_RATIO_MIN'])
                     else 'NO_WINDOW_CONTRAST')
    v['Q9_null_comB'] = S['null_comB_inc']
    v['Q9_null_share'] = S['null_share']
    v['Q9_conf'] = dict(com_B_conf=S['com_B_conf'], com_B_disc=S['com_B'],
                        d_com=({c: (abs(S['com_B_conf'][c] - S['com_B'][c])
                                    if (S['com_B_conf'].get(c) is not None and S['com_B'][c] is not None)
                                    else None) for c in S['com_B_conf']}))
    v['Q10_pert_rel_inc_max'] = float(np.nanmax(S['pert_rel_inc']))
    v['Q10_pert_rel_cum_max'] = float(np.nanmax(S['pert_rel_cum']))
    return v


def joint_verdict(V, recs):
    JV = {}
    arms = list(V.keys())
    JV['arms_present'] = arms
    JV['Q0_apparatus_all'] = all(V[a]['Q0_apparatus'] for a in arms)
    JV['Q0_device_all'] = all(V[a]['Q0_device'] == 'cuda' for a in arms)
    JV['Q1_joint'] = ('FID_ALL_PASS' if all(V[a]['Q1_label'] == 'FID_PASS' for a in arms)
                      else ('FID_PARTIAL' if any(V[a]['Q1_label'] == 'FID_PASS' for a in arms) else 'FID_FAIL'))
    JV['Q2_joint'] = ('ANCHOR_ALL_OK' if all(V[a]['Q2_label'] == 'ANCHOR_OK' for a in arms) else 'ANCHOR_DRIFT')
    JV['Q3_joint'] = ('BRIDGE_ALL_OK' if all(V[a]['Q3_label'] == 'BRIDGE_OK' for a in arms)
                      else ('BRIDGE_PARTIAL' if any(V[a]['Q3_label'] == 'BRIDGE_OK' for a in arms)
                            else 'BRIDGE_DRIFT'))
    nm = sum(1 for a in arms if V[a]['Q4_label'] == 'MLP_DOMINANT_BEH')
    JV['Q4_counts'] = dict(MLP_DOMINANT_BEH=nm, n=len(arms))
    JV['Q4_joint'] = ('MLP_DOMINANT_BEH_ALL' if nm == len(arms)
                      else ('MLP_DOMINANT_BEH_PARTIAL' if nm > 0 else 'MLP_NOT_DOMINANT_BEH_ALL'))
    nc = sum(1 for a in arms if V[a]['Q5_label'] == 'ATTRIBUTION_CONSISTENT')
    JV['Q5_counts'] = dict(CONSISTENT=nc, n=len(arms))
    JV['Q5_joint'] = ('ATTRIBUTION_CONSISTENT_ALL' if nc == len(arms)
                      else ('ATTRIBUTION_CONSISTENT_PARTIAL' if nc > 0 else 'ATTRIBUTION_DISCREPANT_ALL'))
    ns = sum(1 for a in arms if V[a]['Q6_label'] == 'SHALLOWER')
    JV['Q6_counts'] = dict(SHALLOWER=ns, n=len(arms))
    JV['Q6_joint'] = ('BEHAVIOR_SHALLOWER_ALL' if ns == len(arms)
                      else ('BEHAVIOR_SHALLOWER_PARTIAL' if ns > 0 else 'BEHAVIOR_NOT_SHALLOWER'))
    nd = sum(1 for a in arms if V[a]['Q6_deep_label'] == 'DEEP')
    JV['Q6_deep_counts'] = dict(DEEP=nd, n=len(arms))
    JV['Q6_deep_joint'] = ('COMB_DEEP_ALL' if nd == len(arms)
                           else ('COMB_DEEP_PARTIAL' if nd > 0 else 'COMB_SHALLOW_ALL'))
    ncp = sum(1 for a in arms if V[a]['Q7_label'] == 'EFFICACY_COUPLED')
    JV['Q7_counts'] = dict(COUPLED=ncp, n=len(arms))
    JV['Q7_joint'] = ('EFFICACY_COUPLED_ALL' if ncp == len(arms)
                      else ('EFFICACY_COUPLED_PARTIAL' if ncp > 0 else 'EFFICACY_DECOUPLED_ALL'))
    nsw = sum(1 for a in arms if V[a]['Q8_label'] == 'SUPERADD_AT_WINDOW')
    JV['Q8_counts'] = dict(SUPERADD_AT_WINDOW=nsw, n=len(arms))
    JV['Q8_joint'] = ('SUPERADD_AT_WINDOW_ALL' if nsw == len(arms)
                      else ('SUPERADD_AT_WINDOW_PARTIAL' if nsw > 0 else 'NO_WINDOW_CONTRAST_ALL'))
    nh = sum(1 for a in arms if V[a]['Q9_null_comB'].get('com_tail') == 'high')
    JV['Q9_counts'] = dict(null_high=nh, n=len(arms))
    JV['Q9_joint'] = ('NULL_TAIL_HIGH_ALL' if nh == len(arms)
                      else ('NULL_TAIL_HIGH_PARTIAL' if nh > 0 else 'NULL_TAIL_NOT_HIGH'))
    return JV


def predictions_check(JV, V, recs):
    arms = JV['arms_present']
    hold = [a for a in arms if not a.startswith('A0')]
    P = {}
    P['P1'] = dict(pass_=bool(JV['Q0_apparatus_all'] and JV['Q0_device_all']
                              and JV['Q1_joint'] == 'FID_ALL_PASS'),
                   detail=dict(Q0_all=JV['Q0_apparatus_all'], device_all=JV['Q0_device_all'],
                               Q1=JV['Q1_joint'],
                               arch=[V[a]['Q1_arch_max'] for a in arms],
                               blk=[V[a]['Q1_blk_max'] for a in arms]))
    P['P2'] = dict(pass_=bool(JV['Q2_joint'] == 'ANCHOR_ALL_OK' and JV['Q3_joint'] == 'BRIDGE_ALL_OK'),
                   detail=dict(Q2=JV['Q2_joint'], Q3=JV['Q3_joint'],
                               bridge_rel=[V[a]['Q3_bridge_rel'] for a in arms]))
    P['P3'] = dict(name='holdout 主预测 1 —— 组件归属是行为的',
                   pass_=bool(JV['Q4_counts']['MLP_DOMINANT_BEH'] >= 2 and
                              JV['Q5_counts']['CONSISTENT'] >= 2),
                   detail=dict(Q4_counts=JV['Q4_counts'], Q5_counts=JV['Q5_counts'],
                               share_beh={a: V[a]['Q4_share_mlp_beh_nb'] for a in arms},
                               share_vec={a: V[a]['Q4_share_mlp_vec_nb'] for a in arms},
                               holdout=hold))
    P['P4'] = dict(name='holdout 主预测 2 —— 行为质心比向量质心浅',
                   pass_=bool(JV['Q6_counts']['SHALLOWER'] >= len(arms) - 1),
                   detail=dict(Q6_counts=JV['Q6_counts'],
                               com_B={a: V[a]['Q6_com_B'] for a in arms},
                               com_V={a: V[a]['Q6_com_V'] for a in arms},
                               gap={a: V[a]['Q6_gap'] for a in arms},
                               median={a: V[a]['Q6_median_reach'] for a in arms},
                               joint=JV['Q6_joint']))
    P['P5'] = dict(name='同对象耦合 —— 向量质量预测行为效力',
                   pass_=bool(JV['Q7_counts']['COUPLED'] >= 2),
                   detail=dict(Q7_counts=JV['Q7_counts'],
                               sp_wb={a: V[a]['Q7_spearman_wall_ball'] for a in arms},
                               sp_wJ={a: V[a]['Q7_spearman_wall_J'] for a in arms},
                               holdout=hold))
    P['P6'] = dict(name='超可加性峰值落在写入窗',
                   pass_=bool(JV['Q8_counts']['SUPERADD_AT_WINDOW'] >= 2),
                   detail=dict(Q8_counts=JV['Q8_counts'],
                               rlin_argmax={a: V[a]['Q8_rlin_argmax'] for a in arms},
                               L_star_own={a: V[a]['Q8_L_star_own'] for a in arms},
                               rlin_peak_ratio={a: V[a]['Q8_rlin_peak_ratio'] for a in arms},
                               rlin_nb={a: V[a]['Q8_rlin_nb'] for a in arms},
                               rlin_reach={a: V[a]['Q8_rlin_reach'] for a in arms},
                               share_ratio_mlp_attn_nb={a: None for a in arms}))
    P['P7'] = dict(name='对照（置换零假设 + 确认集 + 离流形诊断）',
                   pass_=None,
                   detail=dict(Q9=JV['Q9_joint'], Q9_counts=JV['Q9_counts'],
                               conf={a: V[a]['Q9_conf'] for a in arms},
                               pert_rel_inc={a: V[a]['Q10_pert_rel_inc_max'] for a in arms},
                               pert_rel_cum={a: V[a]['Q10_pert_rel_cum_max'] for a in arms},
                               note='描述性条款：只报告与判定，不设方向性预测。'))
    return P


def main():
    if MERGE:
        recs = {}
        for a in ARM_ORDER:
            p = os.path.join(P18T, '_armrec18_%s.json' % a)
            if os.path.exists(p):
                recs[a] = json.load(io.open(p, encoding='utf-8'))
        V = {a: per_arm_verdict(recs[a]) for a in recs}
        JV = joint_verdict(V, recs)
        PRE = predictions_check(JV, V, recs)
        RES = dict(phase=18, line='N2h1-alpha-11', kind='result', smoke=SMOKE,
                   created_local=time.strftime('%Y-%m-%d %H:%M:%S'),
                   seal_sha256=EX['seal_sha256'], exec_sha256=sha(EXECP),
                   anchor_result_p16_sha256=EX['anchor_result_p16_sha256'],
                   anchor_result_p17_sha256=EX['anchor_result_p17_sha256'],
                   arms=recs, verdict=V, joint_verdict=JV, predictions_check=PRE,
                   floors=FL, bootstrap=EX['bootstrap'], components=COMPONENTS,
                   components_confirmation=COMPONENTS_CONF,
                   bridge_site={a: int(EX['p16_anchors'][a]['L_star_own']) for a in recs},
                   elapsed_total_s=(float(os.environ['ELAPSED_TOTAL'])
                                    if os.environ.get('ELAPSED_TOTAL') else None))
        out = os.path.join(P18T, 'result_phase18%s.json' % ('_smoke' if SMOKE else ''))
        io.open(out, 'w', encoding='utf-8').write(json.dumps(RES, ensure_ascii=False, indent=1))
        w('MERGED -> %s' % out)
        w('joint: %s' % json.dumps({k: JV[k] for k in JV if k.endswith('_joint') or k.endswith('_counts')},
                                   ensure_ascii=False))
        w('predictions: %s' % json.dumps({k: PRE[k]['pass_'] for k in PRE}, ensure_ascii=False))
        return

    sel = [a for a in ARM_ORDER if (not ARMS_SEL) or any(a == s or a.startswith(s + '_') or a.startswith(s)
                                                         for s in ARMS_SEL.split(','))]
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
        dst = os.path.join(P18T, '_armrec18_%s.json' % arm_id)
        io.open(dst, 'w', encoding='utf-8').write(json.dumps(rec, ensure_ascii=False, indent=1))
        w('--- ARM %s DONE %.1fs -> %s' % (arm_id, rec['seconds'], dst))
    if SPLIT_PARTIAL:
        w('SPLIT_PARTIAL: 只写 partial，不合并')
    else:
        w('总耗时 %.1fs（未合并；合并请 MERGE=1）' % (time.time() - t_all))


if __name__ == '__main__':
    main()
