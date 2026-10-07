# -*- coding: utf-8 -*-
"""
Phase 19 (N2h1-alpha-12) 主脚本：写入向量谱 w_ell 与其质心 com_V 的**量化口径稳健性**。

臂（量化口径为唯一自变量；除量化外加载配置逐项一致）：
  A0_nf4  qwen3-4b   nf4   （校准臂 1：逐位复现 P17 锚 com_V=26.1501）
  A0_bf16 qwen3-4b   bf16  （核心检验臂）
  A1_nf4  glm4-9b    nf4   （校准臂 2：逐位复现 P17 锚 com_V=26.7037）
  A1_bf16 glm4-9b    bf16  （跨家族 holdout 检验臂；需 CPU offload）

判据全部预先冻结在 seal 中；本脚本只执行不做判断（判决函数按 seal 的 q 表产出标签）。
用法：SMOKE=1 python n2h1a12_quant_scheme_robustness.py
      SPLIT_PARTIAL=1 ARMS=A0_nf4 python ...
      MERGE=1 python ...
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

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P19 = os.path.join(ROOT, 'tests', 'deepseek', 'Phase19')
P19T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase19')
P17T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase17')
EXECP = os.path.join(P19T, 'execution_phase19.json')
SEALP = os.path.join(P19T, 'N2h1a12_design_seal.json')
ANCHP = os.path.join(P17T, 'result_phase17.json')
SMOKE = os.environ.get('SMOKE', '0') == '1'
ARMS_SEL = [x for x in os.environ.get('ARMS', '').split(',') if x]
SPLIT_PARTIAL = os.environ.get('SPLIT_PARTIAL', '0') == '1'
MERGE = os.environ.get('MERGE', '0') == '1'
os.makedirs(P19T, exist_ok=True)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


EX = json.load(io.open(EXECP, encoding='utf-8'))
assert sha(SEALP) == EX['seal_sha256'], 'DRIFT: seal sha != execution.seal_sha256'
_ab = open(ANCHP, 'rb').read()
assert hashlib.sha256(_ab).hexdigest() == EX['anchor_result_sha256'], 'DRIFT: Phase 17 result 锚漂移'
R17 = json.loads(_ab.decode('utf-8'))

TMPL = EX['template']
SUPS = list(EX['classes'])
INST_ALL = [tuple(x) for x in EX['instances_all']]
PAIRS_ALL = [tuple(x) for x in EX['pairs_all']]
DISC = [tuple(x) for x in EX['discovery']]
CONF = [tuple(x) for x in EX['confirmation']]
DISC_W = set(x[0] for x in DISC)
PROFILE = list(EX['profile_sites'])
QUANT_NF4 = EX['quant_nf4']
QUANT_BF16 = EX['quant_bf16']
NBW = int(EX['neighbourhood_width'])
FL = EX['floors']
BP = int(EX['bootstrap']['BP'])
SEEDS = dict(EX['bootstrap']['seeds'])
ARM_ORDER = list(EX['arm_order'])
ARMS_CFG = EX['arms']
ANCH = EX['anchor_values']
REACH_BY_MODEL = {k: [int(x) for x in v] for k, v in EX['reach_by_model'].items()}

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
    DISC = [d for d in DISC if d[0] in set(x[0] for x in INST_ALL)] or DISC
    DISC_W = set(x[0] for x in DISC)

_log = []


def w(s=''):
    _log.append(str(s))
    print(s)
    sys.stdout.flush()


def F3(v, nd=3):
    return (('%.' + str(nd) + 'f') % v) if isinstance(v, (int, float)) and v is not None else str(v)


# ---------------------------------------------------------------- 统计工具（逐字继承 P17）
def com_of_mass(mass_by_site, sites):
    """区间求和 + 中点质心（P17 逐字同口径）。"""
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
    out = np.full(n_bp, np.nan)
    s = np.asarray(sites, float)
    mid = (s[:-1] + s[1:]) / 2.0
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
    return dict(BP=n_bp, n_ok=int(len(fin)), obs_com=obs, com_p5=p5, com_p95=p95, com_tail=tail, reason=None)


def spearman(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    return float((ra * rb).sum() / den) if den > 1e-12 else None


# ---------------------------------------------------------------- 单臂
def run_arm(arm_id, acfg):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    scheme = acfg['quant']
    rec = dict(arm=arm_id, role=acfg['role'], model=acfg['model'], scheme=scheme,
               offload=bool(acfg.get('offload')), smoke=SMOKE)
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
        # [E-offload] accelerate 的 CPU-offload 用 meta 占位参数承载真实权重；此时参数 device 是 'meta'，
        # 真实执行设备在 module._hf_hook.execution_device。若直接用参数 device 构造输入张量，
        # 会在前向里触发 'Cannot copy out of meta tensor'。
        h = getattr(m, '_hf_hook', None)
        ed = getattr(h, 'execution_device', None) if h is not None else None
        if ed is not None:
            return ed
        for p in m.parameters():
            return p.device
        return torch.device('cuda')

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
    w('  F1b 类别 token: %s' % json.dumps(SUP_ID, ensure_ascii=False))

    tl = {}
    for wd, sup in INST_ALL:
        tl.setdefault(len(ids_of(TMPL % wd)), []).append(wd)
    rec['token_len_hist'] = {str(k): len(v) for k, v in sorted(tl.items())}
    rec['T2_only'] = bool(sorted(tl.keys()) == [2])
    w('  F1 T=2: hist=%s only_T2=%s' % (rec['token_len_hist'], rec['T2_only']))

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
    rec['cfg'] = dict(L=L, hid=HID, n_heads=NH, head_dim=HDP, o_proj_in=OIN,
                      tie=bool(getattr(CFG, 'tie_word_embeddings', False)),
                      vocab=int(CFG.vocab_size), param_devices=dev_hist)
    w('  F4 L=%d hid=%d heads=%d head_dim=%d o_proj_in=%d tie=%s drift=%s'
      % (L, HID, NH, HDP, OIN, bool(getattr(CFG, 'tie_word_embeddings', False)), drift))
    assert rec['F5_o_proj_ok'], 'F5 失败'
    all_cuda = all(k.startswith('cuda') for k in dev_hist)
    rec['Q0_device'] = 'cuda' if all_cuda else ('OFFLOAD' if scheme == 'bf16' else 'MIXED')

    def ids_t(text):
        return torch.tensor([ids_of(text)], device=INDEV)

    n_fw = 0
    # --- E0 装置自检
    with torch.no_grad():
        lga = model(input_ids=ids_t(TMPL % '苹果')).logits[0, -1].float().detach().cpu().numpy()
        lgb = model(input_ids=ids_t(TMPL % '苹果')).logits[0, -1].float().detach().cpu().numpy()
        n_fw += 2
    determinism = float(np.max(np.abs(lga - lgb)))
    rec['E0_selfcheck'] = dict(determinism_maxdiff=determinism, Q0_device=rec['Q0_device'],
                              F4_dims_ok=rec['F4_dims_ok'], F5_o_proj_ok=rec['F5_o_proj_ok'])
    w('  E0 determinism=%.3e device=%s' % (determinism, rec['Q0_device']))

    # --- E1 capture（HH / o_proj 输入 / MLP 输出）
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
        return HH, so, sm

    t0 = time.time()
    CAP = {}
    for wd, sup in INST_ALL:
        CAP[wd] = capture(TMPL % wd)
    n_fw += len(INST_ALL)
    rec['E1_capture'] = dict(n=len(CAP), seconds=round(time.time() - t0, 1),
                             hidden_levels=int(CAP[INST_ALL[0][0]][0].shape[0]))
    w('  E1 capture %d 实例 / %.1fs (levels=%d)' % (len(CAP), rec['E1_capture']['seconds'],
                                                    rec['E1_capture']['hidden_levels']))
    HH0 = CAP[INST_ALL[0][0]][0]
    assert HH0.shape[1] == HID and not np.isnan(HH0).any(), 'capture NaN/shape'
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
        t = torch.tensor(blocks, device=mod_dev(opmod), dtype=torch.bfloat16)
        o = opmod(t).float().cpu().numpy()
        return o[:NH], o[NH]

    # --- E2 保真度门
    arch, blk = [], []
    for wd, sup in INST_ALL[:6]:
        HHs, O, M = CAP[wd]
        for l in range(L - 1):
            lhs = HHs[l + 1] - HHs[l]
            with torch.no_grad():
                rhs = OPR[l](torch.tensor(O[l][None, :], device=mod_dev(OPR[l]),
                                          dtype=torch.bfloat16)).detach().float().cpu().numpy()[0] + M[l]
            arch.append(float(np.linalg.norm(lhs - rhs)) / max(float(np.linalg.norm(lhs)), 1e-9))
    for wd, sup in INST_ALL[:4]:
        _, O, M = CAP[wd]
        for l in range(0, L - 1, max(L // 8, 1)):
            hb, full = head_blocks(OPR[l], O[l])
            blk.append(float(np.linalg.norm(hb.sum(0) - full)) / max(float(np.linalg.norm(full)), 1e-9))
    arch = np.array(arch)
    blk = np.array(blk)
    rec['E2_fidelity'] = dict(arch_max=float(arch.max()), arch_mean=float(arch.mean()),
                              blk_max=float(blk.max()), blk_mean=float(blk.mean()),
                              n_arch=int(len(arch)), n_blk=int(len(blk)))
    w('  E2 fidelity arch max=%.3e mean=%.3e | blk max=%.3e mean=%.3e'
      % (arch.max(), arch.mean(), blk.max(), blk.mean()))

    # --- E3 类子空间 U_l
    by = {}
    for wd, sup in INST_ALL:
        by.setdefault(sup, []).append(wd)
    AVAIL = [s for s in SUPS if s in by]
    assert len(AVAIL) >= 2, 'AVAIL<2 => U_l 不可估'
    rk = max(len(AVAIL) - 1, 1)
    U = {}
    for l in range(L):
        mus = np.stack([np.mean([CAP[wd][0][l + 1] for wd in by[s]], 0) for s in AVAIL], 0).astype(np.float64)
        D = mus - mus.mean(0, keepdims=True)
        _, sv, Vt = np.linalg.svd(D, full_matrices=False)
        U[l] = Vt[:rk].astype(np.float32)
    rec['E3_U'] = dict(rank=rk, n_classes=len(AVAIL), classes=AVAIL)
    w('  E3 U_l: rank=%d n_classes=%d' % (rk, len(AVAIL)))

    def proj(v, Ub):
        return (v @ Ub.T) @ Ub

    # --- E4 质量谱
    disc_pairs = [p for p in PAIRS_ALL if p[0] in DISC_W]
    w('  pairing: discovery %d' % len(disc_pairs))

    def mass_profile(pairs):
        A = np.zeros(L - 1)
        AT = np.zeros(L - 1)
        ML = np.zeros(L - 1)
        TP = np.zeros(L - 1)
        DN = np.zeros(L - 1)
        for (rw, rs, dw, ds, sw) in pairs:
            _, OR_, MR = CAP[rw]
            _, OD, MD = CAP[dw]
            for l in range(L - 1):
                Ub = U[l]
                hbD, fullD = head_blocks(OPR[l], OD[l])
                hbR, fullR = head_blocks(OPR[l], OR_[l])
                d_attn = fullD - fullR
                d_mlp = MD[l] - MR[l]
                A[l] += float(np.linalg.norm(proj(d_attn + d_mlp, Ub)))
                AT[l] += float(np.linalg.norm(proj(d_attn, Ub)))
                ML[l] += float(np.linalg.norm(proj(d_mlp, Ub)))
                DN[l] += float(np.linalg.norm(d_attn + d_mlp))
                dd = hbD - hbR
                per = np.zeros(NH)
                for h in range(NH):
                    per[h] = float(np.linalg.norm(proj(dd[h], Ub)))
                TP[l] += float(per.max())
        n = max(len(pairs), 1)
        return A / n, AT / n, ML / n, TP / n, DN / n

    t0 = time.time()
    wA, wAT, wML, wTP, wDN = mass_profile(disc_pairs)
    rec['E4_profile_s'] = round(time.time() - t0, 2)
    w('  E4 profile done / %.2fs' % rec['E4_profile_s'])

    reach = REACH_BY_MODEL[acfg['dir']]
    RE_L = [s for s in reach if 0 <= s < (L - 1)]

    def mass_dict(arr):
        return {int(l): float(arr[l]) for l in range(len(arr))}

    mA, mAT, mML, mTP = mass_dict(wA), mass_dict(wAT), mass_dict(wML), mass_dict(wTP)
    com_all, _ = com_of_mass(mA, RE_L)
    com_mlp, _ = com_of_mass(mML, RE_L)
    com_attn, _ = com_of_mass(mAT, RE_L)
    com_top, _ = com_of_mass(mTP, RE_L)
    com_full, _ = com_of_mass(mA, [s for s in PROFILE if 0 <= s < (L - 1)])
    med_reach = float(np.median(RE_L))
    nb = [l for l in RE_L if abs(l - com_all) <= NBW] if com_all is not None else []
    s_all_nb = float(sum(mA[l] for l in nb))
    share_mlp_nb = (float(sum(mML[l] for l in nb)) / s_all_nb) if s_all_nb > 1e-12 else None
    share_attn_nb = (float(sum(mAT[l] for l in nb)) / s_all_nb) if s_all_nb > 1e-12 else None
    rec['E5_com_V'] = dict(
        reach=RE_L, median_reach=med_reach, com_V=com_all, com_V_mlp=com_mlp,
        com_V_attn=com_attn, com_V_top1head=com_top, com_V_full=com_full,
        neighbourhood=nb, share_mlp_nb=share_mlp_nb, share_attn_nb=share_attn_nb,
        w_all=[float(x) for x in wA], w_attn=[float(x) for x in wAT],
        w_mlp=[float(x) for x in wML], w_top1=[float(x) for x in wTP],
        d_norm=[float(x) for x in wDN],
        argmax_w_layer=(int(np.argmax(wA)) if np.isfinite(wA).all() else None))
    w('  E5 com_V(all)=%s mlp=%s attn=%s top1=%s ; nb=%s share_mlp_nb=%s'
      % (F3(com_all), F3(com_mlp), F3(com_attn), F3(com_top), nb, F3(share_mlp_nb)))
    w('     median(REACH)=%.1f ; com_V_full=%s ; argmax_w @L%s'
      % (med_reach, F3(com_full), rec['E5_com_V']['argmax_w_layer']))

    # --- E6 置换零假设
    rgA = np.random.default_rng(int(SEEDS['comv_all']))
    rgM = np.random.default_rng(int(SEEDS['comv_mlp']))
    rec['E6_null'] = dict(all=perm_null_com(mA, RE_L, rgA, BP),
                          mlp=perm_null_com(mML, RE_L, rgM, BP))
    w('  E6 null all obs=%s p5=%s p95=%s tail=%s | mlp tail=%s'
      % (F3(rec['E6_null']['all']['obs_com']), F3(rec['E6_null']['all']['com_p5']),
         F3(rec['E6_null']['all']['com_p95']), rec['E6_null']['all']['com_tail'],
         rec['E6_null']['mlp']['com_tail']))

    # --- E7 P17 锚逐位断言（仅 nf4 校准臂）
    if scheme == 'nf4':
        an = ANCH[arm_id]
        detail = {}
        ok_all = True
        for key, got, exp_v in [
            ('com_V', com_all, an['com_V']),
            ('com_V_mlp', com_mlp, an['com_V_mlp']),
            ('com_V_attn', com_attn, an['com_V_attn']),
            ('median_reach', med_reach, an['median_reach']),
        ]:
            ok = (got is not None and abs(got - exp_v) <= FL['CALIB_TOL_COMV'])
            detail[key] = dict(got=got, expected=exp_v, ok=bool(ok))
            ok_all = ok_all and bool(ok)
        for key, got, exp_v in [
            ('nb', [int(x) for x in nb], [int(x) for x in an['nb']]),
            ('argmax_w_layer', rec['E5_com_V']['argmax_w_layer'], an['argmax_w_layer']),
            ('share_mlp_nb_round3', round(share_mlp_nb, 3), round(an['share_mlp_nb'], 3)),
        ]:
            ok = (got == exp_v)
            detail[key] = dict(got=got, expected=exp_v, ok=bool(ok))
            ok_all = ok_all and bool(ok)
        rec['E7_anchor'] = dict(ok=bool(ok_all), detail=detail)
        w('  E7 P17 锚复现: %s' % ('OK' if ok_all else 'DRIFT'))
        if not ok_all:
            for k, v in detail.items():
                if not v['ok']:
                    w('     !! %s got=%s expected=%s' % (k, v['got'], v['expected']))
    else:
        rec['E7_anchor'] = dict(ok=None, detail={'note': 'bf16 臂不复现 nf4 锚（按设计）'})

    rec['n_forwards'] = int(n_fw)
    rec['seconds'] = None
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
                             and E0['determinism_maxdiff'] <= 1e-6)
    v['Q1_label'] = ('FID_PASS' if (FID['arch_max'] <= FL['P19_FID_ARCH']
                                    and FID['blk_max'] <= FL['P19_FID_BLK']) else 'FID_FAIL')
    v['Q1_arch_max'] = FID['arch_max']
    v['Q1_blk_max'] = FID['blk_max']
    if rec.get('E7_anchor', {}).get('ok') is None:
        v['Q2_label'] = 'ANCHOR_NA'
    else:
        v['Q2_label'] = ('CALIB_OK' if rec['E7_anchor']['ok'] else 'CALIB_DRIFT')
    v['Q2_detail'] = rec.get('E7_anchor', {}).get('detail')
    v['com_V'] = C5['com_V']
    v['com_V_mlp'] = C5['com_V_mlp']
    v['com_V_attn'] = C5['com_V_attn']
    v['median_reach'] = C5['median_reach']
    v['neighbourhood'] = C5['neighbourhood']
    v['share_mlp_nb'] = C5['share_mlp_nb']
    v['argmax_w_layer'] = C5['argmax_w_layer']
    v['w_all'] = C5['w_all']
    v['Q5_label'] = ('MLP_DOMINANT' if (C5['share_mlp_nb'] is not None
                                        and C5['share_mlp_nb'] > FL['MLP_DOM_MIN'])
                     else 'MLP_NOT_DOMINANT')
    v['Q6_label'] = ('DEEP' if (C5['com_V'] is not None and C5['com_V'] >= C5['median_reach'])
                     else 'SHALLOW')
    v['Q7_null_all'] = rec['E6_null']['all']
    v['Q7_null_mlp'] = rec['E6_null']['mlp']
    return v


PAIRS_QUANT = [('A0_nf4', 'A0_bf16'), ('A1_nf4', 'A1_bf16')]


def quant_pair_stats(V):
    """同 Phase 配对：量化口径敏感度。"""
    out = {}
    for an, ab in PAIRS_QUANT:
        if an not in V or ab not in V:
            continue
        a, b = V[an], V[ab]
        wa = np.asarray(a['w_all'], float)
        wb = np.asarray(b['w_all'], float)
        n = min(len(wa), len(wb))
        wa, wb = wa[:n], wb[:n]
        rel = np.abs(wa - wb) / np.maximum(np.abs(wb), 1e-9)
        out['%s|%s' % (an, ab)] = dict(
            pair=[an, ab], model_nf4=a['Q0_device'],
            com_V_nf4=a['com_V'], com_V_bf16=b['com_V'],
            delta_com_V=abs(a['com_V'] - b['com_V']),
            delta_com_V_mlp=abs(a['com_V_mlp'] - b['com_V_mlp']),
            delta_com_V_attn=abs(a['com_V_attn'] - b['com_V_attn']),
            spearman_w=spearman(wa, wb),
            median_rel_resid=float(np.median(rel)), p90_rel_resid=float(np.percentile(rel, 90)),
            argmax_nf4=a['argmax_w_layer'], argmax_bf16=b['argmax_w_layer'],
            argmax_same=bool(a['argmax_w_layer'] == b['argmax_w_layer']),
            share_mlp_nb_nf4=a['share_mlp_nb'], share_mlp_nb_bf16=b['share_mlp_nb'],
            share_same_side=bool((a['share_mlp_nb'] > FL['MLP_DOM_MIN'])
                                 == (b['share_mlp_nb'] > FL['MLP_DOM_MIN'])),
            nb_nf4=a['neighbourhood'], nb_bf16=b['neighbourhood'])
    return out


def joint_verdict(V, recs):
    JV = {}
    arms = list(V.keys())
    JV['arms_present'] = arms
    JV['Q0_apparatus_all'] = all(V[a]['Q0_apparatus'] for a in arms)
    JV['Q1_joint'] = ('FID_ALL_PASS' if all(V[a]['Q1_label'] == 'FID_PASS' for a in arms)
                      else ('FID_PARTIAL' if any(V[a]['Q1_label'] == 'FID_PASS' for a in arms) else 'FID_FAIL'))
    nf4 = [a for a in arms if a.endswith('_nf4')]
    JV['Q2_joint'] = ('CALIB_ALL_OK' if (nf4 and all(V[a]['Q2_label'] == 'CALIB_OK' for a in nf4))
                      else ('CALIB_DRIFT' if nf4 else 'CALIB_NA'))
    QP = quant_pair_stats(V)
    JV['quant_pairs'] = QP
    stable = [k for k, s in QP.items() if s['delta_com_V'] <= FL['QUANT_TOL_COMV']]
    JV['Q3_counts'] = dict(STABLE=len(stable), n=len(QP))
    JV['Q3_joint'] = ('QUANT_STABLE_ALL' if (QP and len(stable) == len(QP))
                      else ('QUANT_STABLE_PARTIAL' if stable else 'QUANT_SENSITIVE'))
    rho_ok = [k for k, s in QP.items() if (s['spearman_w'] is not None
                                           and s['spearman_w'] >= FL['RHO_SHAPE_MIN'])]
    JV['Q4_counts'] = dict(CONSISTENT=len(rho_ok), n=len(QP))
    JV['Q4_joint'] = ('SPECTRUM_CONSISTENT_ALL' if (QP and len(rho_ok) == len(QP))
                      else ('SPECTRUM_CONSISTENT_PARTIAL' if rho_ok else 'SPECTRUM_DISTORTED'))
    bf = [a for a in arms if a.endswith('_bf16')]
    nm = sum(1 for a in bf if V[a]['Q5_label'] == 'MLP_DOMINANT')
    JV['Q5_counts'] = dict(MLP_DOMINANT=nm, n=len(bf))
    JV['Q5_joint'] = ('MLP_DOM_RETAINED_ALL' if (bf and nm == len(bf))
                      else ('MLP_DOM_RETAINED_PARTIAL' if nm > 0 else 'MLP_DOM_LOST_ALL'))
    nd = sum(1 for a in bf if V[a]['Q6_label'] == 'DEEP')
    JV['Q6_counts'] = dict(DEEP=nd, n=len(bf))
    JV['Q6_joint'] = ('DEEP_RETAINED_ALL' if (bf and nd == len(bf))
                      else ('DEEP_RETAINED_PARTIAL' if nd > 0 else 'DEEP_LOST_ALL'))
    JV['Q7_null'] = {a: dict(all=V[a]['Q7_null_all']['com_tail'], mlp=V[a]['Q7_null_mlp']['com_tail'])
                     for a in arms}
    return JV


def predictions_check(JV, V, recs):
    arms = JV['arms_present']
    bf = [a for a in arms if a.endswith('_bf16')]
    P = {}
    P['P1'] = dict(pass_=bool(JV['Q0_apparatus_all'] and JV['Q1_joint'] == 'FID_ALL_PASS'
                              and JV['Q2_joint'] == 'CALIB_ALL_OK'),
                   detail=dict(Q0=JV['Q0_apparatus_all'], Q1=JV['Q1_joint'], Q2=JV['Q2_joint'],
                               calib={a: V[a]['Q2_label'] for a in arms if a.endswith('_nf4')}))
    kA0 = 'A0_nf4|A0_bf16'
    s0 = JV['quant_pairs'].get(kA0)
    P['P2'] = dict(pass_=bool(s0 and s0['delta_com_V'] <= FL['QUANT_TOL_COMV']
                              and V['A0_bf16']['Q6_label'] == 'DEEP'
                              and s0['spearman_w'] is not None
                              and s0['spearman_w'] >= FL['RHO_SHAPE_MIN']),
                   detail=s0)
    kA1 = 'A1_nf4|A1_bf16'
    s1 = JV['quant_pairs'].get(kA1)
    P['P3'] = dict(pass_=bool(s1 and s1['delta_com_V'] <= FL['QUANT_TOL_COMV']), detail=s1)
    P['P4'] = dict(pass_=bool('A1_bf16' in V and V['A1_bf16']['Q5_label'] == 'MLP_DOMINANT'
                              and V['A1_bf16']['Q6_label'] == 'DEEP'),
                   detail=dict(share_mlp_nb=(V['A1_bf16']['share_mlp_nb'] if 'A1_bf16' in V else None),
                               com_V=(V['A1_bf16']['com_V'] if 'A1_bf16' in V else None),
                               median=(V['A1_bf16']['median_reach'] if 'A1_bf16' in V else None)))
    rho_all = [s['spearman_w'] for s in JV['quant_pairs'].values()]
    P['P5'] = dict(pass_=bool(rho_all and all(x is not None and x >= FL['RHO_SHAPE_MIN'] for x in rho_all)),
                   detail=dict(rho=rho_all,
                               resid={k: dict(med=s['median_rel_resid'], p90=s['p90_rel_resid'])
                                      for k, s in JV['quant_pairs'].items()}))
    return P


def main():
    t_start = time.time()
    w('=' * 78)
    w('Phase 19 | N2h1-alpha-12 | SMOKE=%s ARMS=%s SPLIT=%s MERGE=%s' % (SMOKE, ARMS_SEL, SPLIT_PARTIAL, MERGE))
    w('  seal=%s exec=%s' % (sha(SEALP)[:8], sha(EXECP)[:8]))

    if MERGE:
        recs = {}
        for a in ARM_ORDER:
            p = os.path.join(P19T, '_armrec19_%s.json' % a)
            if os.path.exists(p):
                recs[a] = json.load(io.open(p, encoding='utf-8'))
        V = {a: per_arm_verdict(recs[a]) for a in recs}
        JV = joint_verdict(V, recs)
        PC = predictions_check(JV, V, recs)
        OUT = dict(phase=19, line='N2h1-alpha-12', kind='result', smoke=SMOKE,
                   created_local=time.strftime('%Y-%m-%d %H:%M:%S'),
                   seal_sha256=sha(SEALP), exec_sha256=sha(EXECP),
                   anchor_result_sha256=sha(ANCHP),
                   arms=recs, verdict=V, joint_verdict=JV, predictions_check=PC,
                   floors=FL, bootstrap=EX['bootstrap'],
                   reach_by_model=REACH_BY_MODEL,
                   elapsed_total_s=(float(os.environ['ELAPSED_TOTAL'])
                                    if os.environ.get('ELAPSED_TOTAL')
                                    else round(time.time() - t_start, 1)))
        name = 'result_phase19_smoke.json' if SMOKE else 'result_phase19.json'
        o = os.path.join(P19T, name)
        io.open(o, 'w', encoding='utf-8').write(json.dumps(OUT, ensure_ascii=False, indent=1))
        w('MERGE -> %s (%d arms)' % (os.path.basename(o), len(recs)))
        for k in sorted(PC):
            w('  %s pass=%s' % (k, PC[k]['pass_']))
        w('  Q2=%s Q3=%s Q4=%s Q5=%s Q6=%s' % (JV['Q2_joint'], JV['Q3_joint'], JV['Q4_joint'],
                                               JV['Q5_joint'], JV['Q6_joint']))
        for k, s in sorted(JV['quant_pairs'].items()):
            w('  pair %s: delta_com_V=%.4f rho=%s argmax(N/B)=%s/%s share(N/B)=%.4f/%.4f'
              % (k, s['delta_com_V'], F3(s['spearman_w'], 4), s['argmax_nf4'], s['argmax_bf16'],
                 s['share_mlp_nb_nf4'], s['share_mlp_nb_bf16']))
        return

    sel = ARMS_SEL if ARMS_SEL else ARM_ORDER
    for arm_id in sel:
        if SPLIT_PARTIAL:
            acfg = ARMS_CFG[arm_id]
            t_arm = time.time()
            w('-' * 70)
            w('[ARM %s] %s (%s)' % (arm_id, acfg['model'], acfg['quant']))
            rec = run_arm(arm_id, acfg)
            rec['seconds'] = round(time.time() - t_arm, 1)
            o = os.path.join(P19T, '_armrec19_%s.json' % arm_id)
            io.open(o, 'w', encoding='utf-8').write(json.dumps(rec, ensure_ascii=False, indent=1))
            w('[ARM %s] done %.1fs -> %s' % (arm_id, rec['seconds'], os.path.basename(o)))
        else:
            w('[ARM %s] (single-process mode; 建议用 SPLIT_PARTIAL)' % arm_id)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
