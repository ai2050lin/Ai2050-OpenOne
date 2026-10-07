# -*- coding: utf-8 -*-
"""
Phase 17 (N2h1-alpha-10) 主脚本：写入向量的位置与效力。

三臂（同一 nf4 口径，串行/进程隔离）：
  A0_calib_qwen3-4b-nf4 : 校准/装置臂（可行性探针所在臂）
  A1_glm4-9b-nf4        : 跨家族 untied 复算（holdout）
  A2_qwen3-14b-nf4      : 同家族 untied 规模放大复算（holdout + 判别臂）

本 Phase 相对 Phase 16 的**唯一**改动：把 Phase 8 的**向量预算 share_v**（精确可加）从单层 L6
推广到**逐层**，得到**向量质量谱 w_ell**，并定义其质心 com_V，与 P16 的两个**行为**质心
com_layer(xhalf/J) 三向对照。

判定全部预先冻结在 seal 中；本脚本只执行不做判断（判决函数按 seal 的 q 表产出标签）。
用法：SMOKE=1 python n2h1a10_writevec_centroid.py        (A0，小配对网格)
      SPLIT_PARTIAL=1 ARMS=A0 python ...                 (逐臂进程隔离)
      MERGE=1 python ...                                 (合并三臂 -> result)
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
P17 = os.path.join(ROOT, 'tests', 'deepseek', 'Phase17')
P17T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase17')
P16T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase16')
EXECP = os.path.join(P17T, 'execution_phase17.json')
SEALP = os.path.join(P17T, 'N2h1a10_design_seal.json')
ANCHP = os.path.join(P16T, 'result_phase16.json')
SMOKE = os.environ.get('SMOKE', '0') == '1'
ARMS_SEL = os.environ.get('ARMS', '')
SPLIT_PARTIAL = os.environ.get('SPLIT_PARTIAL', '0') == '1'
MERGE = os.environ.get('MERGE', '0') == '1'
os.makedirs(P17T, exist_ok=True)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def sha8(p):
    return sha(p)[:8]


EX = json.load(io.open(EXECP, encoding='utf-8'))
assert sha(SEALP) == EX['seal_sha256'], 'DRIFT: seal sha != execution.seal_sha256'

_ab = open(ANCHP, 'rb').read()
assert hashlib.sha256(_ab).hexdigest() == EX['anchor_result_sha256'], 'DRIFT: Phase 16 result 锚漂移'
ANCH_ALL = json.loads(_ab.decode('utf-8'))
ANCH = EX['anchor_values']

TMPL = EX['template']
SUPS = list(EX['classes'])
INST_ALL = [tuple(x) for x in EX['instances_all']]
PAIRS_ALL = [tuple(x) for x in EX['pairs_all']]
DISC = [tuple(x) for x in EX['discovery']]
CONF = [tuple(x) for x in EX['confirmation']]
DISC_W = set(x[0] for x in DISC)
CONF_W = set(x[0] for x in CONF)
PROFILE = list(EX['profile_sites'])
QUANT = EX['quant']
KS = list(EX['span_ks'])
NBW = int(EX['neighbourhood_width'])
FL = EX['floors']
BP = int(EX['bootstrap']['BP'])
SEEDS = dict(EX['bootstrap']['seeds'])
ARM_ORDER = list(EX['arm_order'])
ARMS_CFG = EX['arms']

if SMOKE:
    # SMOKE 必须让「配对集非空且实例集覆盖配对的两端」——否则 U_l 与质量谱全退化
    # （首次 SMOKE 实测：只截前 8 实例 + 前 6 配对 -> discovery 0 对 -> com_V=None -> 崩溃）。
    BP = 200
    PAIRS_ALL = PAIRS_ALL[:8]
    _keep = []
    for _p in PAIRS_ALL:
        for _x in (_p[0], _p[2]):
            if _x not in _keep:
                _keep.append(_x)
    _sel = [t for t in INST_ALL if t[0] in _keep]
    INST_ALL = _sel if len(_sel) >= 4 else INST_ALL[:8]
    DISC_W = set(x[0] for x in DISC)
    CONF_W = set(x[0] for x in CONF)
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
    """与 Phase 16 逐字节同口径：mid = 相邻位点中点，com = sum|j|*mid / sum|j|。"""
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
    """把逐层质量按**相邻位点区间**聚合 W_j = sum_{l in [s_j, s_{j+1})} w_l，再取质心（与 stat_com_layer 同 mid）。

    ⚠️ 必须是**区间求和**，不是「位点单层取值」—— seal 原文如此，且探针正是这么算的。
    （首版实现取了 sites[j] 单层，导致与探针 A0 的 com_V 差 1.68 层；已作同轮勘误 E4。）
    """
    s = np.asarray(sites, float)
    vals = np.asarray([float(sum(mass_by_site.get(int(l), 0.0)
                                for l in range(int(sites[j]), int(sites[j + 1]))))
                       for j in range(len(sites) - 1)], float)
    mid = (s[:-1] + s[1:]) / 2.0
    den = float(vals.sum())
    if not np.isfinite(den) or den <= 1e-12:
        return None, None
    return float((vals * mid).sum() / den), vals


def stat_span_k(jumps, k):
    j = np.asarray(jumps, float)
    n = len(j)
    if n < k or not np.isfinite(j).all():
        return None
    idx = np.argsort(-np.abs(j))[:k]
    return float(idx.max() - idx.min()) / max(n - 1, 1)


def perm_null_com(mass_by_site, sites, rng_obj, n_bp):
    """com_V 的置换零假设：保留质量多重集，随机重排到位点（顺序敏感 => 非退化）。"""
    obs, vals = com_of_mass(mass_by_site, sites)
    if vals is None:
        return dict(BP=n_bp, n_ok=0, obs_com=None, com_p5=None, com_p95=None,
                    com_tail=None, reason='empty_or_zero_mass')
    n = len(vals)
    out = np.full(n_bp, np.nan)
    for b in range(n_bp):
        p = vals[rng_obj.permutation(n)]
        s = np.asarray(sites, float)
        mid = (s[:-1] + s[1:]) / 2.0
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


def perm_null_span(jumps, rng_obj, n_bp, k):
    j = np.asarray(jumps, float)
    n = len(j)
    if n < k + 1 or not np.isfinite(j).all():
        return dict(BP=n_bp, n_ok=0, obs_span=None, span_p5=None, span_p95=None,
                    span_tail=None, reason='bad_input')
    obs = stat_span_k(j, k)
    ss = np.array([stat_span_k(j[rng_obj.permutation(n)], k) for _ in range(n_bp)], float)
    fin = ss[np.isfinite(ss)]
    if len(fin) == 0:
        return dict(BP=n_bp, n_ok=0, obs_span=obs, span_p5=None, span_p95=None,
                    span_tail=None, reason='all_nan')
    p5, p95 = float(np.percentile(fin, 5)), float(np.percentile(fin, 95))
    tail = ('low' if (obs is not None and obs <= p5) else
            'high' if (obs is not None and obs >= p95) else 'none')
    return dict(BP=n_bp, n_ok=int(len(fin)), obs_span=obs, span_p5=p5, span_p95=p95,
                span_tail=tail, degenerate=bool(p95 - p5 < 1e-12), reason=None)


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = len(a)
    if n < 3:
        return None
    if float(np.std(a)) <= 1e-9 or float(np.std(b)) <= 1e-9:
        return None          # 退化输入（常量向量）不得产出伪相关
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

    # --- F1b 类别 token 逐臂现场解析（Phase 15 amend1 事故的修复条款）
    SUP_ID = {}
    for _wd in SUPS:
        _t = list(tok.encode(_wd, add_special_tokens=False))
        assert len(_t) == 1 and tok.decode([_t[0]]) == _wd, \
            'F1b 失败：类别词 %r 非单 token 或 decode 不可逆 %r' % (_wd, _t)
        SUP_ID[_wd] = int(_t[0])
    rec['sup_id_arm'] = dict(SUP_ID)
    rec['F1b_ok'] = True
    w('  F1b 类别 token（逐臂解析）: %s' % json.dumps(SUP_ID, ensure_ascii=False))

    # --- F1 T=2 布局
    tl = {}
    for wd, sup in INST_ALL:
        tl.setdefault(len(ids_of(TMPL % wd)), []).append(wd)
    rec['token_len_hist'] = {str(k): len(v) for k, v in sorted(tl.items())}
    rec['T2_only'] = bool(sorted(tl.keys()) == [2])
    w('  F1 T=2 布局: hist=%s only_T2=%s' % (rec['token_len_hist'], rec['T2_only']))

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

    def ids_t(text):
        return torch.tensor([ids_of(text)], device='cuda')

    # --- E0 装置自检
    n_fw = 0
    with torch.no_grad():
        lga = model(input_ids=ids_t(TMPL % '苹果')).logits[0, -1].float().detach().cpu().numpy()
        lgb = model(input_ids=ids_t(TMPL % '苹果')).logits[0, -1].float().detach().cpu().numpy()
        n_fw += 2
    determinism = float(np.max(np.abs(lga - lgb)))
    rng0 = np.random.default_rng(20261003)
    vec0 = (rng0.standard_normal(HID) * 0.1).astype(np.float32)
    site0 = L // 2

    @torch.no_grad()
    def fwd_patch(text, site, vec):
        ii = ids_t(text)
        mod = layers[site]

        def hook(m, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            t = t.clone()
            t[0, -1, :] = vec.to(t.dtype)
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

    # --- E1 capture（扩展：HH / o_proj 输入 / MLP 输出 / logits）
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
                             hidden_levels=int(CAP[INST_ALL[0][0]][0].shape[0]),
                             has_O=True, has_M=True)
    w('  E1 capture %d 实例 / %.1fs (hidden levels=%d; O/M 已存)'
      % (len(CAP), rec['E1_capture']['seconds'], rec['E1_capture']['hidden_levels']))
    HH0 = CAP[INST_ALL[0][0]][0]
    assert HH0.shape[1] == HID and not np.isnan(HH0).any(), 'SMOKE/capture NaN or shape'
    assert CAP[INST_ALL[0][0]][1][L - 1].shape[0] == OIN

    # --- 逐头分块（模块自身前向；不用权重矩阵）
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

    # --- 类子空间 U_l（level=l+1，与 P16 E3 同口径：全实例按类平均）
    by = {}
    for wd, sup in INST_ALL:
        by.setdefault(sup, []).append(wd)
    AVAIL = [s for s in SUPS if s in by]
    rk = max(len(AVAIL) - 1, 1)
    U = {}
    for l in range(L):
        mus = np.stack([np.mean([CAP[wd][0][l + 1] for wd in by[s]], 0) for s in AVAIL], 0).astype(np.float64)
        D = mus - mus.mean(0, keepdims=True)
        _, sv, Vt = np.linalg.svd(D, full_matrices=False)
        U[l] = Vt[:rk].astype(np.float32)
    rec['E3_U'] = dict(rank=rk, n_classes=len(AVAIL), classes=AVAIL)
    w('  E3 U_ell: rank=%d n_classes=%d' % (rk, len(AVAIL)))

    def proj(v, Ub):
        return (v @ Ub.T) @ Ub

    # --- E4 向量质量谱（逐层组件预算）
    disc_pairs = [p for p in PAIRS_ALL if p[0] in DISC_W and p[0] in CAP and p[2] in CAP]
    conf_pairs = [p for p in PAIRS_ALL if p[0] in CONF_W and p[0] in CAP and p[2] in CAP]
    w('  pairing: discovery %d / confirmation %d' % (len(disc_pairs), len(conf_pairs)))

    def mass_profile(pairs):
        A = np.zeros(L - 1); AT = np.zeros(L - 1); ML = np.zeros(L - 1); TP = np.zeros(L - 1)
        HD = np.zeros((NH, L - 1))
        for (rw, rs, dw, ds, sw) in pairs:
            _, OR_, MR, _ = CAP[rw]
            _, OD, MD, _ = CAP[dw]
            for l in range(L - 1):
                Ub = U[l]
                hbD, fullD = head_blocks(OPR[l], OD[l])
                hbR, fullR = head_blocks(OPR[l], OR_[l])
                d_attn = fullD - fullR
                d_mlp = MD[l] - MR[l]
                A[l] += float(np.linalg.norm(proj(d_attn + d_mlp, Ub)))
                AT[l] += float(np.linalg.norm(proj(d_attn, Ub)))
                ML[l] += float(np.linalg.norm(proj(d_mlp, Ub)))
                dd = hbD - hbR
                per = np.zeros(NH)
                for h in range(NH):
                    per[h] = float(np.linalg.norm(proj(dd[h], Ub)))
                    HD[h, l] += per[h]
                TP[l] += float(per.max())
        n = max(len(pairs), 1)
        return A / n, AT / n, ML / n, TP / n, HD / n

    t0 = time.time()
    wA, wAT, wML, wTP, wHD = mass_profile(disc_pairs)
    rec['E4_profile_s'] = round(time.time() - t0, 2)
    w('  E4 向量质量谱完成 / %.2fs' % rec['E4_profile_s'])

    reach = [int(x) for x in ANCH[arm_id]['reach']]
    sites_p16 = [int(x) for x in ANCH[arm_id]['sites']]
    J_p16 = {int(s): float(v) for s, v in zip(sites_p16, ANCH[arm_id]['J_by_site'])}
    xh_p16 = {int(s): float(v) for s, v in zip(sites_p16, ANCH_ALL['E4_summary'][arm_id]['xhalf'])}
    RE_L = [s for s in reach if 0 <= s < (L - 1)]

    def mass_dict(arr):
        return {int(l): float(arr[l]) for l in range(len(arr))}

    mA = mass_dict(wA); mAT = mass_dict(wAT); mML = mass_dict(wML); mTP = mass_dict(wTP)

    com_all, vec_all = com_of_mass(mA, RE_L)
    com_mlp, _ = com_of_mass(mML, RE_L)
    com_attn, _ = com_of_mass(mAT, RE_L)
    com_top, _ = com_of_mass(mTP, RE_L)
    com_full, _ = com_of_mass(mA, [s for s in PROFILE if 0 <= s < (L - 1)])
    med_reach = float(np.median(RE_L))

    # 邻域组件归属
    nb = [l for l in RE_L if abs(l - com_all) <= NBW] if com_all is not None else []
    s_all_nb = float(sum(mA[l] for l in nb))
    s_mlp_nb = float(sum(mML[l] for l in nb))
    s_att_nb = float(sum(mAT[l] for l in nb))
    share_mlp_nb = (s_mlp_nb / s_all_nb) if s_all_nb > 1e-12 else None
    share_attn_nb = (s_att_nb / s_all_nb) if s_all_nb > 1e-12 else None
    hd_nb = np.array([sum(wHD[h][l] for l in nb) for h in range(NH)]) if nb else np.zeros(NH)
    top1_share_nb = (float(hd_nb.max()) / s_all_nb) if (nb and s_all_nb > 1e-12) else None

    rec['E5_com_V'] = dict(
        reach=RE_L, median_reach=med_reach, com_V=com_all, com_V_mlp=com_mlp,
        com_V_attn=com_attn, com_V_top1head=com_top, com_V_full=com_full,
        neighbourhood=nb, share_mlp_nb=share_mlp_nb, share_attn_nb=share_attn_nb,
        top1_head_nb=(int(np.argmax(hd_nb)) if nb else None), top1_head_share_nb=top1_share_nb,
        w_all=[float(x) for x in wA], w_attn=[float(x) for x in wAT],
        w_mlp=[float(x) for x in wML], w_top1=[float(x) for x in wTP],
        head_mass=[float(hd_nb[h]) for h in range(NH)],
        argmax_w_layer=(int(np.argmax(wA)) if np.isfinite(wA).all() else None))
    w('  E5 com_V(all)=%s mlp=%s attn=%s top1=%s ; 邻域=%s ; share_mlp_nb=%s'
      % (F3(com_all), F3(com_mlp), F3(com_attn), F3(com_top), nb, F3(share_mlp_nb)))
    w('     median(REACH)=%.1f ; com_V_full=%s ; argmax w @L%s'
      % (med_reach, F3(com_full), rec['E5_com_V']['argmax_w_layer']))

    # 确认集复核
    wAc, _, wMLc, _, _ = mass_profile(conf_pairs) if conf_pairs else (None, None, None, None, None)
    if wAc is not None:
        com_all_c, _ = com_of_mass(mass_dict(wAc), RE_L)
        com_mlp_c, _ = com_of_mass(mass_dict(wMLc), RE_L)
    else:
        com_all_c = com_mlp_c = None
    rec['E5b_conf'] = dict(n_pairs=len(conf_pairs), com_V=com_all_c, com_V_mlp=com_mlp_c,
                           d_com=(abs(com_all_c - com_all) if (com_all_c is not None and com_all is not None) else None))
    w('  E5b 确认集 n=%d: com_V=%s (delta=%s)'
      % (len(conf_pairs), F3(com_all_c), F3(rec['E5b_conf']['d_com'])))

    # --- E6 效力关系 spearman(w_ell, J_ell)
    xl = [l for l in RE_L if l in J_p16]
    sp_wJ = spearman([mA[l] for l in xl], [J_p16[l] for l in xl])
    sp_wx = spearman([mA[l] for l in xl], [xh_p16[l] for l in xl]) if all(l in xh_p16 for l in xl) else None
    sp_Jdepth = spearman([J_p16[l] for l in xl], xl)
    rec['E6_efficacy'] = dict(n=len(xl), sites=xl, spearman_wJ=sp_wJ, spearman_wxhalf=sp_wx,
                              spearman_Jdepth=sp_Jdepth,
                              w_at_sites=[float(mA[l]) for l in xl],
                              J_at_sites=[float(J_p16[l]) for l in xl])
    w('  E6 spearman(w,J)=%s ; spearman(w,xhalf)=%s ; spearman(J,depth)=%s (n=%d)'
      % (F3(sp_wJ, 4), F3(sp_wx, 4), F3(sp_Jdepth, 4), len(xl)))

    # --- E7 置换零假设（com_V 两条线）
    rgA = np.random.default_rng(int(SEEDS['comv_all']))
    rgM = np.random.default_rng(int(SEEDS['comv_mlp']))
    rec['E7_null'] = dict(all=perm_null_com(mA, RE_L, rgA, BP),
                          mlp=perm_null_com(mML, RE_L, rgM, BP))
    w('  E7 置换零假设: all obs=%s p5=%s p95=%s tail=%s ; mlp obs=%s tail=%s'
      % (F3(rec['E7_null']['all']['obs_com']), F3(rec['E7_null']['all']['com_p5']),
         F3(rec['E7_null']['all']['com_p95']), rec['E7_null']['all']['com_tail'],
         F3(rec['E7_null']['mlp']['obs_com']), rec['E7_null']['mlp']['com_tail']))

    # --- E8 span_k 谱（行为剖面：xhalf / J，k in KS）
    jumps = {c: np.diff(np.array([ (J_p16 if c == 'j' else xh_p16)[l] for l in RE_L], float))
             for c in ('x', 'j')}
    spans = {}
    for c in ('x', 'j'):
        spans[c] = {}
        for k in KS:
            rg = np.random.default_rng(int(SEEDS['comv_all']) + 1000 * k + (0 if c == 'x' else 1))
            spans[c][str(k)] = perm_null_span(jumps[c], rg, BP, k)
        spans[c]['com_layer'] = stat_com_layer(jumps[c], RE_L)
    prec_x = stat_com_layer(jumps['x'], RE_L)
    prec_j = stat_com_layer(jumps['j'], RE_L)
    rec['E8_span'] = dict(sites=RE_L, spans=spans, com_layer_x=prec_x, com_layer_j=prec_j)
    for c in ('x', 'j'):
        w('  E8 span %s: %s' % (c, ' '.join('k%d=%s' % (k, F3(spans[c][str(k)]['obs_span'], 4)) for k in KS)))

    # --- E9 P16 锚逐位断言
    an = ANCH[arm_id]
    anchor_ok = True
    anchor_detail = {}
    for key, got, exp in [
        ('com_layer_x', prec_x, an['com_layer_x']),
        ('com_layer_j', prec_j, an['com_layer_j']),
        ('L_star_own', ANCH_ALL['E3_localize'][arm_id]['L_star_own'], an['L_star_own']),
        ('ell_reach', ANCH_ALL['E7_reach'][arm_id]['ell_reach'], an['ell_reach']),
        ('reach', [int(x) for x in ANCH_ALL['E7_reach'][arm_id]['reach']], an['reach']),
    ]:
        if key in ('com_layer_x', 'com_layer_j'):
            ok = (got is not None and abs(got - exp) <= 1e-6)
        else:
            ok = (got == exp)
        anchor_detail[key] = dict(got=got, expected=exp, ok=bool(ok))
        anchor_ok = anchor_ok and bool(ok)
    # span3 锚（P16 的 new_stat obs_span，k=3）
    for c, key in (('x', 'span3_x'), ('j', 'span3_j')):
        got = spans[c]['3']['obs_span']
        exp = an[key]
        ok = (got is not None and abs(got - exp) <= 1e-6)
        anchor_detail[key] = dict(got=got, expected=exp, ok=bool(ok))
        anchor_ok = anchor_ok and bool(ok)
    rec['E9_anchor'] = dict(ok=bool(anchor_ok), detail=anchor_detail)
    w('  E9 P16 锚复现: %s' % ('OK' if anchor_ok else 'DRIFT'))
    if not anchor_ok:
        for k, v in anchor_detail.items():
            if not v['ok']:
                w('     !! %s got=%s expected=%s' % (k, v['got'], v['expected']))

    rec['n_forwards'] = int(n_fw)
    rec['sup_id_matches_ref'] = None
    rec['cands_used'] = []
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rec


# ---------------------------------------------------------------- 判决
def per_arm_verdict(rec):
    E0 = rec['E0_selfcheck']
    FID = rec['E2_fidelity']
    C5 = rec['E5_com_V']
    v = {}
    v['Q0_device'] = rec['Q0_device']
    v['Q0_apparatus'] = bool(rec['T2_only'] and rec['F4_dims_ok'] and rec['F5_o_proj_ok']
                             and E0['determinism_maxdiff'] <= 1e-6 and E0['hook_effect_maxdiff'] > 1e-6)
    v['Q1_label'] = ('FID_PASS' if (FID['arch_max'] <= FL['P17_FID_ARCH']
                                    and FID['blk_max'] <= FL['P17_FID_BLK']) else 'FID_FAIL')
    v['Q1_arch_max'] = FID['arch_max']
    v['Q1_blk_max'] = FID['blk_max']
    v['Q2_label'] = ('ANCHOR_OK' if rec['E9_anchor']['ok'] else 'ANCHOR_DRIFT')
    v['Q2_detail'] = rec['E9_anchor']['detail']
    comv = C5['com_V']
    v['Q3_com_V'] = comv
    v['Q3_median'] = C5['median_reach']
    v['Q3_label'] = ('DEEP' if (comv is not None and comv >= C5['median_reach']) else 'SHALLOW')
    dx = abs(comv - ANCH[rec['arm']]['com_layer_x']) if comv is not None else None
    dj = abs(comv - ANCH[rec['arm']]['com_layer_j']) if comv is not None else None
    v['Q4_d_x'] = dx
    v['Q4_d_j'] = dj
    v['Q4_min_d'] = min(dx, dj) if (dx is not None and dj is not None) else None
    v['Q4_label'] = ('TRANSFORM_ALIGNED' if (v['Q4_min_d'] is not None
                                             and v['Q4_min_d'] < FL['CENTROID_SEP_MIN'])
                     else 'POSITION_DECOUPLED')
    v['Q5_share_mlp_nb'] = C5['share_mlp_nb']
    v['Q5_label'] = ('MLP_DOMINANT' if (C5['share_mlp_nb'] is not None
                                        and C5['share_mlp_nb'] > FL['MLP_DOM_MIN'])
                     else 'MLP_NOT_DOMINANT')
    sp = rec['E6_efficacy']['spearman_wJ']
    v['Q6_spearman_wJ'] = sp
    v['Q6_label'] = ('WRITE_EFFICACY_ANTICORR' if (sp is not None and sp < 0)
                     else 'WRITE_EFFICACY_COUPLED')
    v['Q7_null_all'] = rec['E7_null']['all']
    v['Q7_null_mlp'] = rec['E7_null']['mlp']
    v['Q7_span'] = {c: {k: rec['E8_span']['spans'][c][k]['obs_span'] for k in rec['E8_span']['spans'][c] if k != 'com_layer'}
                    for c in ('x', 'j')}
    v['Q8_conf'] = rec['E5b_conf']
    return v


def joint_verdict(V, recs):
    JV = {}
    arms = list(V.keys())
    JV['arms_present'] = arms
    JV['Q0_apparatus_all'] = all(V[a]['Q0_apparatus'] for a in arms)
    JV['Q0_device_all'] = all(V[a]['Q0_device'] == 'cuda' for a in arms)
    JV['Q1_joint'] = ('FID_ALL_PASS' if all(V[a]['Q1_label'] == 'FID_PASS' for a in arms)
                      else ('FID_PARTIAL' if any(V[a]['Q1_label'] == 'FID_PASS' for a in arms)
                            else 'FID_FAIL'))
    JV['Q2_joint'] = ('ANCHOR_ALL_OK' if all(V[a]['Q2_label'] == 'ANCHOR_OK' for a in arms)
                      else 'ANCHOR_DRIFT')
    nd = sum(1 for a in arms if V[a]['Q3_label'] == 'DEEP')
    JV['Q3_counts'] = dict(DEEP=nd, n=len(arms))
    JV['Q3_joint'] = ('DEEP_ALL' if nd == len(arms) else ('DEEP_PARTIAL' if nd > 0 else 'DEEP_NONE'))
    ndc = sum(1 for a in arms if V[a]['Q4_label'] == 'POSITION_DECOUPLED')
    JV['Q4_counts'] = dict(DECOUPLED=ndc, n=len(arms))
    JV['Q4_joint'] = ('POSITION_DECOUPLED_ALL' if ndc == len(arms)
                      else ('POSITION_DECOUPLED_PARTIAL' if ndc > 0 else 'TRANSFORM_ALIGNED_ALL'))
    nm = sum(1 for a in arms if V[a]['Q5_label'] == 'MLP_DOMINANT')
    JV['Q5_counts'] = dict(MLP_DOMINANT=nm, n=len(arms))
    JV['Q5_joint'] = ('MLP_DOMINANT_ALL' if nm == len(arms)
                      else ('MLP_DOMINANT_PARTIAL' if nm > 0 else 'MLP_NOT_DOMINANT_ALL'))
    na = sum(1 for a in arms if V[a]['Q6_label'] == 'WRITE_EFFICACY_ANTICORR')
    JV['Q6_counts'] = dict(ANTICORR=na, n=len(arms))
    JV['Q6_joint'] = ('WRITE_EFFICACY_ANTICORR_ALL' if na == len(arms)
                      else ('WRITE_EFFICACY_ANTICORR_PARTIAL' if na > 0 else 'WRITE_EFFICACY_COUPLED_ALL'))
    # span 序 vs com_layer 序（x 相对 J）：seal 原文是「是否**同号**」。
    # v1 实现误把配对写成 (sx<sj)==(cx>cj)（"x 更窄 AND x 更深"，这是一个**错位配对**，
    # 不等于"同号"）=> 3/3 报 DECOUPLED。v2 修正为同一比较方向 (sx>sj)==(cx>cj)。
    # 统计量与数据**未变**，仅标签语义修正；两条读数都保留（同轮勘误留痕）。
    coup, coup_v1 = [], []
    for a in arms:
        rec = recs[a]
        sp = rec['E8_span']['spans']
        try:
            sx = float(sp['x']['3']['obs_span']); sj = float(sp['j']['3']['obs_span'])
            cx = float(sp['x']['com_layer']); cj = float(sp['j']['com_layer'])
            coup.append(bool((sx > sj) == (cx > cj)))        # 同号（x 更大 span 与 x 更深 com 一致）
            coup_v1.append(bool((sx < sj) == (cx > cj)))     # v1 的错位配对（留痕）
        except Exception:
            coup.append(None); coup_v1.append(None)
    JV['Q7_coupled'] = coup
    JV['Q7_coupled_v1_mismatched_pairing'] = coup_v1
    JV['Q7_joint'] = ('SPAN_CENTROID_COUPLED' if all(c is True for c in coup)
                      else ('SPAN_CENTROID_DECOUPLED' if all(c is False for c in coup)
                            else 'SPAN_CENTROID_MIXED'))
    JV['Q7_joint_v1_mismatched_pairing'] = ('SPAN_CENTROID_COUPLED' if all(c is True for c in coup_v1)
                                            else ('SPAN_CENTROID_DECOUPLED' if all(c is False for c in coup_v1)
                                                  else 'SPAN_CENTROID_MIXED'))
    return JV


def predictions_check(JV, V, recs):
    arms = JV['arms_present']
    P = {}
    P['P1'] = dict(pass_=bool(JV['Q0_apparatus_all'] and JV['Q0_device_all']
                              and JV['Q1_joint'] == 'FID_ALL_PASS'),
                   detail=dict(Q0_all=JV['Q0_apparatus_all'], device_all=JV['Q0_device_all'],
                               Q1=JV['Q1_joint'],
                               arch=[V[a]['Q1_arch_max'] for a in arms],
                               blk=[V[a]['Q1_blk_max'] for a in arms]))
    P['P2'] = dict(pass_=bool(JV['Q2_joint'] == 'ANCHOR_ALL_OK'), detail=dict(Q2=JV['Q2_joint']))
    hold = [a for a in arms if not a.startswith('A0')]
    deep_h = [V[a]['Q3_label'] for a in hold]
    P['P3'] = dict(pass_=bool(all(x == 'DEEP' for x in deep_h) and len(hold) >= 2),
                   detail=dict(holdout=hold, labels=deep_h,
                               com_V={a: V[a]['Q3_com_V'] for a in arms},
                               median={a: V[a]['Q3_median'] for a in arms},
                               joint=JV['Q3_joint']))
    a2 = [a for a in arms if a.startswith('A2')]
    if a2:
        a2 = a2[0]
        P['P4'] = dict(pass_=bool(V[a2]['Q4_label'] == 'POSITION_DECOUPLED'),
                       detail=dict(discriminator=a2, label=V[a2]['Q4_label'],
                                   min_d=V[a2]['Q4_min_d'], d_x=V[a2]['Q4_d_x'], d_j=V[a2]['Q4_d_j'],
                                   all_min_d={a: V[a]['Q4_min_d'] for a in arms},
                                   joint=JV['Q4_joint']))
    else:
        P['P4'] = dict(pass_=None, detail=dict(reason='A2 absent'))
    P['P5'] = dict(pass_=bool(JV['Q5_counts']['MLP_DOMINANT'] >= 2), detail=dict(JV['Q5_counts']))
    P['P6'] = dict(pass_=bool(JV['Q6_counts']['ANTICORR'] >= 2),
                   detail=dict(JV['Q6_counts'], sp={a: V[a]['Q6_spearman_wJ'] for a in arms}))
    P['P7'] = dict(pass_=None, detail=dict(joint=JV['Q7_joint'], coupled=JV['Q7_coupled'],
                                           span={a: V[a]['Q7_span'] for a in arms},
                                           note='描述性条款：只报告与判定，不设方向性预测'))
    return P


def main():
    if MERGE:
        recs = {}
        for a in ARM_ORDER:
            p = os.path.join(P17T, '_armrec17_%s.json' % a)
            if os.path.exists(p):
                recs[a] = json.load(io.open(p, encoding='utf-8'))
        V = {a: per_arm_verdict(recs[a]) for a in recs}
        JV = joint_verdict(V, recs)
        PRE = predictions_check(JV, V, recs)
        RES = dict(phase=17, line='N2h1-alpha-10', kind='result', smoke=SMOKE,
                   created_local=time.strftime('%Y-%m-%d %H:%M:%S'),
                   seal_sha256=EX['seal_sha256'], exec_sha256=sha(EXECP),
                   anchor_result_sha256=EX['anchor_result_sha256'],
                   arms=recs, verdict=V, joint_verdict=JV, predictions_check=PRE,
                   floors=FL, bootstrap=EX['bootstrap'], span_ks=KS,
                   elapsed_total_s=(float(os.environ['ELAPSED_TOTAL'])
                                    if os.environ.get('ELAPSED_TOTAL') else None))
        out = os.path.join(P17T, 'result_phase17%s.json' % ('_smoke' if SMOKE else ''))
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
        dst = os.path.join(P17T, '_armrec17_%s.json' % arm_id)
        io.open(dst, 'w', encoding='utf-8').write(json.dumps(rec, ensure_ascii=False, indent=1))
        w('--- ARM %s DONE %.1fs -> %s' % (arm_id, rec['seconds'], dst))
    if SPLIT_PARTIAL:
        w('SPLIT_PARTIAL: 只写 partial，不合并')
    else:
        w('总耗时 %.1fs（未合并；合并请 MERGE=1）' % (time.time() - t_all))


if __name__ == '__main__':
    main()
