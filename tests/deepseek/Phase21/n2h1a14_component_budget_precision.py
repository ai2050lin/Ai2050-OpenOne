# -*- coding: utf-8 -*-
"""
Phase 21 (N2h1-alpha-14) 主脚本：
  组件级向量预算 `share_v`（M1，第一指标，精确可加）与 权重实现级容量 `W`（M2）的
  **跨精度稳健性**（bitsandbytes nf4 ↔ torch.bfloat16）。

动机：P8（N2h1-alpha）在 **bf16** 下算出 L6 写入算子的 share_v（MLP 0.4717 / 最大单头 head14 0.0742）
与 W 容量；而 P16/P17/P18 把同一预算推广时**全在 nf4 口径**。P19 只补向量侧、P20 只补行为/剖面侧。
本 Phase 在同一装置内把 P8 线上的**两个原始量**在 nf4 与 bf16 下复算 ⇒ 闭合谱系精度缺口。

关键：`share_v` 是**向量预算**——**不需要额外前向**（只需 capture + U + 投影）。
      M3（效应侧 dDonor）为**第二指标**，需要前向；用 NOFX=1 可跳过。

用法：
  SMOKE=1 SPLIT_PARTIAL=1 ARMS=A0_nf4 python ...      # 冒烟
  SPLIT_PARTIAL=1 ARMS=A0_nf4 python ...              # 逐臂（进程隔离）
  MERGE=1 python ...                                  # 合并 -> result_phase21.json
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
P21T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase21')
EXECP = os.path.join(P21T, 'execution_phase21.json')
SEALP = os.path.join(P21T, 'N2h1a14_design_seal.json')
RESULT = os.path.join(P21T, 'result_phase21.json')

SMOKE = os.environ.get('SMOKE', '0') == '1'
ARMS_SEL = os.environ.get('ARMS', '')
SPLIT_PARTIAL = os.environ.get('SPLIT_PARTIAL', '0') == '1'
MERGE = os.environ.get('MERGE', '0') == '1'
NOFX = os.environ.get('NOFX', '0') == '1'

os.makedirs(P21T, exist_ok=True)
os.makedirs(os.path.join(P21T, 'smoke'), exist_ok=True)

_log = []


def w(s=''):
    _log.append(str(s))
    print(s)
    sys.stdout.flush()


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def F3(v, nd=3):
    return ('%.' + str(nd) + 'f') % v if isinstance(v, (int, float)) and v is not None else str(v)


EX = json.load(io.open(EXECP, encoding='utf-8'))
assert sha(SEALP) == EX['seal_sha256'], 'DRIFT: seal sha != execution.seal_sha256'
EXEC_SHA8 = sha(EXECP)[:8]
SEAL_SHA8 = sha(SEALP)[:8]

P8R = os.path.join(ROOT, EX['anchor_p8_result_path'])
assert sha(P8R) == EX['anchor_p8_result_sha256'], 'DRIFT: P8 result 锚漂移'
A8 = json.load(io.open(P8R, encoding='utf-8'))
AP8 = EX['anchors']['A0_p8_bf16']

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
FL = EX['floors']
ARM_ORDER = list(EX['arm_order'])
ARMS_CFG = EX['arms']
VRP = int(EX['V_rand_per_pair'])
SEED = int(EX['seed'])

if SMOKE:
    # 取跨 4 个不同 rw 类的 4 对（保证 U 的类数 >= 4；SMOKE 下秩退化为 3，仅验管线）
    _pick = []; _seen = set()
    for _p in PAIRS_ALL:
        if _p[1] not in _seen:
            _seen.add(_p[1]); _pick.append(_p)
        if len(_pick) >= 4:
            break
    PAIRS_ALL = _pick
    _w = set()
    for _p in _pick:
        _w.add(_p[0]); _w.add(_p[2])
    INST_ALL = [t for t in INST_ALL if t[0] in _w]
    DISC = [d for d in DISC if d[0] in _w]
    CONF = []


# ---------------------------------------------------------------- 统计工具
def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) != len(b) or len(a) < 3:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    if d <= 1e-12:
        return None
    return float((ra * rb).sum() / d)


def proj(vec, Ub):
    return (vec @ Ub.T) @ Ub


def score_rank(v, sup, sid, sup_id):
    v = np.array(v, copy=True)
    v[sid] = -1e9
    own = float(v[sup_id[sup]])
    others = [float(v[sup_id[x]]) for x in SUPS if x != sup]
    return own - float(np.mean(others))


def score_rank_full(v, sup, sid, sup_id):
    v = np.array(v, copy=True)
    v[sid] = -1e9
    own = float(v[sup_id[sup]])
    others = [float(v[sup_id[x]]) for x in SUPS if x != sup]
    order = np.argsort(-v)
    return own - float(np.mean(others)), int(np.where(order == sup_id[sup])[0][0]) + 1


# ---------------------------------------------------------------- 臂执行
def run_arm(arm_id, acfg):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    scheme = acfg['quant']
    rec = dict(arm=arm_id, role=acfg['role'], model=acfg['model'], scheme=scheme,
               offload=bool(acfg.get('offload')), smoke=SMOKE, nofx=NOFX,
               primary_layer=int(acfg['primary_layer']), L_star=int(acfg['L_star_own']))
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
        assert len(_t) == 1 and tok.decode([_t[0]]) == _wd, 'F1b: 类词 %r 非单 token/不可逆 %r' % (_wd, _t)
        SUP_ID[_wd] = int(_t[0])
    rec['sup_id_arm'] = dict(SUP_ID)
    rec['F1b_ok'] = True

    tl = {}
    for wd, sup in INST_ALL:
        tl.setdefault(len(ids_of(TMPL % wd)), []).append(wd)
    rec['token_len_hist'] = {str(k): len(v) for k, v in sorted(tl.items())}
    rec['T2_only'] = bool(sorted(tl.keys()) == [2])
    w('  F1 T=2: hist=%s only=%s' % (rec['token_len_hist'], rec['T2_only']))

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
    w('  F4: L=%d hid=%d heads=%d head_dim=%d o_proj_in=%d(%s) tie=%s drift=%s'
      % (L, HID, NH, HDP, OIN, onm, bool(getattr(CFG, 'tie_word_embeddings', False)), drift))
    assert rec['F5_o_proj_ok'], 'F5 失败：o_proj_in %d != NH*HD %d' % (OIN, NH * HDP)
    assert not drift, 'F4 drift=%s' % drift
    all_cuda = all(k.startswith('cuda') for k in dev_hist)
    rec['Q0_device'] = 'cuda' if all_cuda else ('OFFLOAD' if scheme == 'bf16' else 'MIXED')

    PRI = int(acfg['primary_layer'])
    PRE = PRI - 1
    assert 1 <= PRE < PRI <= L - 2, 'primary 层越界 PRI=%d L=%d' % (PRI, L)
    HEADN = ['head%d' % h for h in range(NH)]
    DENOM = HEADN + ['mlp']
    NAMES = HEADN + ['mlp', 'diff5', 'attn_all', 'diff6']
    w('  写入窗 L%d（PRE=L%d）; heads=%d ; DENOM=%d' % (PRI, PRE, NH, len(DENOM)))

    def ids_t(text):
        return torch.tensor([ids_of(text)], device=INDEV)

    n_fw = 0

    @torch.no_grad()
    def fwd_plain(text):
        return model(input_ids=ids_t(text))

    with torch.no_grad():
        lga = model(input_ids=ids_t(TMPL % INST_ALL[0][0])).logits[0, -1].float().detach().cpu().numpy()
        lgb = model(input_ids=ids_t(TMPL % INST_ALL[0][0])).logits[0, -1].float().detach().cpu().numpy()
        n_fw += 2
    determinism = float(np.max(np.abs(lga - lgb)))

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

    rng0 = np.random.default_rng(SEED)
    vec0 = (rng0.standard_normal(HID) * 0.1).astype(np.float32)
    lg_hook = fwd_patch(TMPL % INST_ALL[0][0], L // 2, vec0)
    n_fw += 1
    hook_effect = float(np.max(np.abs(lg_hook - lgb)))
    rec['E0_selfcheck'] = dict(determinism_maxdiff=determinism, hook_effect_maxdiff=hook_effect,
                              Q0_device=rec['Q0_device'], F4_dims_ok=rec['F4_dims_ok'])
    w('  E0: determinism=%.3e ; hook effect=%.3e ; device=%s' % (determinism, hook_effect, rec['Q0_device']))

    # ---------------- E1 capture（只取所需位点）----------------
    @torch.no_grad()
    def capture(text):
        ii = ids_t(text)
        so, sm = {}, {}
        hs = []

        def ho(m, args):
            so[PRI] = args[0].detach()[0, -1, :].float().cpu().numpy().copy()

        def hm(m, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            sm[PRI] = t.detach()[0, -1, :].float().cpu().numpy().copy()
        hs.append(OPR[PRI].register_forward_pre_hook(ho))
        hs.append(MLPS[PRI].register_forward_hook(hm))
        out = model(input_ids=ii, output_hidden_states=True)
        for h in hs:
            h.remove()
        HH = out.hidden_states
        hh = {PRE + 1: HH[PRE + 1][0, -1].float().detach().cpu().numpy().copy(),
              PRI + 1: HH[PRI + 1][0, -1].float().detach().cpu().numpy().copy()}
        return hh, so[PRI], sm[PRI], out.logits[0, -1].float().detach().cpu().numpy()

    t_cap = time.time()
    CAP = {}
    for wd, sup in INST_ALL:
        CAP[wd] = capture(TMPL % wd)
    n_fw += len(INST_ALL)
    w('  capture %d instances in %.1fs' % (len(CAP), time.time() - t_cap))
    _w0 = INST_ALL[0][0]
    assert CAP[_w0][1].shape[0] == OIN and CAP[_w0][2].shape[0] == HID and CAP[_w0][0][PRI + 1].shape[0] == HID
    assert not np.isnan(CAP[_w0][0][PRI + 1]).any()
    PAIRS = [p for p in PAIRS_ALL if p[0] in CAP and p[2] in CAP]
    w('  usable pairs = %d / %d' % (len(PAIRS), len(PAIRS_ALL)))

    # ---------------- E2 U（P8 口径：只用 discovery）----------------
    by_class = {}
    for wd, sup in DISC:
        by_class.setdefault(sup, []).append(wd)
    AVAIL = [s for s in SUPS if s in by_class]
    assert len(AVAIL) >= 4, 'U 估计：discovery 类数 %d < 4' % len(AVAIL)
    mus = np.stack([np.mean([CAP[wd][0][PRI + 1] for wd in by_class[s]], 0) for s in AVAIL], 0).astype(np.float64)
    Dm = mus - mus.mean(0, keepdims=True)
    _, sv, Vt = np.linalg.svd(Dm, full_matrices=False)
    RK = max(len(AVAIL) - 1, 1)
    U = Vt[:RK].astype(np.float32)
    rec['U'] = dict(n_classes=len(AVAIL), n_disc=len(DISC), rank=int(RK),
                    sing=[float(x) for x in sv[:RK]], primary_layer=PRI)
    w('  U: n_classes=%d n=%d rank=%d sing=%s' % (len(AVAIL), len(DISC), RK, ' '.join('%.1f' % x for x in sv[:RK])))

    AO = U  # [5, HID]

    # ---------------- 组件增量 ----------------
    W_O = None

    def get_weightT(mod):
        """y = W^T（[in_f, out_f]）：喂 I_{in_f} 过模块，绕开 bnb 反量化内部。"""
        in_f = int(mod.in_features)
        dev = mod_dev(mod)
        eye = torch.eye(in_f, dtype=torch.bfloat16, device=dev)
        with torch.no_grad():
            y = mod(eye)
        y = y[0] if isinstance(y, tuple) else y
        return y.detach().float().cpu().numpy()

    def comp_deltas(rw, dw_):
        h_PRr = CAP[rw][0][PRE + 1].astype(np.float32)
        h_PRd = CAP[dw_][0][PRE + 1].astype(np.float32)
        h_Pr = CAP[rw][0][PRI + 1].astype(np.float32)
        h_Pd = CAP[dw_][0][PRI + 1].astype(np.float32)
        m_r = CAP[rw][2].astype(np.float32)
        m_d = CAP[dw_][2].astype(np.float32)
        o_r = CAP[rw][1].astype(np.float32)
        o_d = CAP[dw_][1].astype(np.float32)
        d = {'diff5': h_PRd - h_PRr, 'mlp': m_d - m_r}
        hd = np.zeros_like(h_Pr)
        for h in range(NH):
            sl = slice(h * HDP, (h + 1) * HDP)
            da = W_O[:, sl] @ (o_d[sl] - o_r[sl])
            d['head%d' % h] = da
            hd += da
        d['attn_all'] = hd
        d['diff6'] = h_Pd - h_Pr
        return d

    # 权重矩阵：W_O = (W^T).T（[HID, OIN]）
    t_w = time.time()
    W_OT = get_weightT(OPR[PRI])              # [OIN, HID]
    W_O = W_OT.T.copy()                        # [HID, OIN]
    rec['WOT_shape'] = list(W_OT.shape)
    w('  W_O via identity-probe: %s -> W_O%s (%.1fs)' % (W_OT.shape, W_O.shape, time.time() - t_w))

    # ---------------- E3/E4 M1 向量预算（第一指标，无前向）----------------
    VEC = {k: [0.0, 0] for k in NAMES}
    DISCP = [p for p in PAIRS if p[0] in DISC_W]
    for (rw, rs, dw_, ds, sw) in DISCP:
        D = comp_deltas(rw, dw_)
        for nm in NAMES:
            pv = proj(D[nm].astype(np.float32), AO)
            VEC[nm][0] += float(np.linalg.norm(pv)); VEC[nm][1] += 1
    VECT = {k: (v[0] / v[1] if v[1] else 0.0) for k, v in VEC.items()}
    vb = {k: VECT[k] for k in DENOM}
    tot_v = sum(vb.values())
    share_v = {k: (v / tot_v if tot_v > 0 else 0.0) for k, v in vb.items()}
    maxv = max(share_v[k] for k in HEADN)
    argv = max(HEADN, key=lambda k: share_v[k])
    mlp_share_v = share_v['mlp']
    loo_vec_top1 = 1.0 - maxv
    rec['M1'] = dict(vec_budget=dict(vb), vec_budget_total=float(tot_v), share_v=dict(share_v),
                     max_head_share_v=float(maxv), argmax_head_v=str(argv),
                     share_v_mlp=float(mlp_share_v), loo_vec_top1=float(loo_vec_top1),
                     vec_budget_diff5=float(VECT['diff5']), vec_budget_attn_all=float(VECT['attn_all']),
                     vec_budget_diff6=float(VECT['diff6']),
                     ratio_diff5_diff6=float(VECT['diff5'] / max(VECT['diff6'], 1e-12)),
                     n_pairs=len(DISCP))
    rk = sorted(DENOM, key=lambda k: -share_v[k])[:6]
    w('  [M1] share_v top6: %s' % ', '.join('%s=%.4f' % (k, share_v[k]) for k in rk))
    w('  [M1] mlp_share_v=%.6f max_head_share_v=%.6f(%s) loo_vec_top1=%.6f'
      % (mlp_share_v, maxv, argv, loo_vec_top1))
    w('  [M1] ||P_U(diff5)||=%.3f / ||P_U(diff6)||=%.3f = %.3f'
      % (VECT['diff5'], VECT['diff6'], VECT['diff5'] / max(VECT['diff6'], 1e-12)))

    # ---------------- E5 M2 权重容量（无前向）----------------
    try:
        cap = W_OT @ AO.T                       # [OIN, 5] ; block h = (AO @ W_O[:,h])^T
        per_head = np.array([float((cap[h * HDP:(h + 1) * HDP, :] ** 2).sum()) for h in range(NH)])
        down = layers[PRI].mlp.down_proj
        WdT = get_weightT(down)                 # [inter, HID]
        mlp_cap = float(((WdT @ AO.T) ** 2).sum())
        tot_c = float(per_head.sum())
        hs = (per_head / tot_c) if tot_c > 0 else per_head * 0
        rec['M2'] = dict(head_share=[float(x) for x in hs], max_head_share=float(hs.max()),
                         argmax_head=int(np.argmax(per_head)),
                         mlp_share_vs_attn=float(mlp_cap / max(tot_c, 1e-12)),
                         attn_cap_total=tot_c, mlp_cap=mlp_cap)
        w('  [M2] W: max_head_share=%.6f(#%d) mlp_share_vs_attn=%.6f'
          % (hs.max(), int(np.argmax(per_head)), mlp_cap / max(tot_c, 1e-12)))
    except Exception as e:
        rec['M2'] = dict(error='%s: %s' % (type(e).__name__, e))
        w('  [M2] FAILED: %s' % e)

    # ---------------- E6/E7 M3 效应侧（第二指标，需前向）----------------
    if NOFX:
        rec['M3'] = dict(skipped=True)
    else:
        BASE = {}
        for (rw, rs, dw_, ds, sw) in PAIRS:
            sr0, rr0 = score_rank_full(CAP[rw][3], rs, ids_of(rw)[0], SUP_ID)
            sd0, rd0 = score_rank_full(CAP[rw][3], ds, ids_of(dw_)[0], SUP_ID)
            BASE[rw] = dict(sr0=sr0, sd0=sd0, rr0=rr0, rd0=rd0)
        bad_base = [rw for rw, b in BASE.items() if not (b['sr0'] > 0)]
        rec['base_ok'] = (not bad_base)
        rec['base_n'] = len(BASE)
        T = {nm: [0.0, 0, 0] for nm in NAMES}
        T['V_rand'] = [0.0, 0, 0]
        T['M_mismatch'] = [0.0, 0, 0]
        rng = np.random.default_rng(SEED + 17)
        for (rw, rs, dw_, ds, sw) in DISCP:
            B = BASE[rw]; sid_r = ids_of(rw)[0]; sid_d = ids_of(dw_)[0]
            h_Pr = CAP[rw][0][PRI + 1].astype(np.float32)
            D = comp_deltas(rw, dw_)
            for nm in NAMES:
                pv = proj(D[nm].astype(np.float32), AO)
                v1 = fwd_patch(TMPL % rw, PRI, h_Pr + pv)
                T[nm][0] += score_rank(v1, ds, sid_d, SUP_ID) - B['sd0']
                T[nm][1] += score_rank(v1, rs, sid_r, SUP_ID) - B['sr0']
                T[nm][2] += 1
            if sw is not None and sw in CAP:
                Dm = comp_deltas(rw, sw)
                v1 = fwd_patch(TMPL % rw, PRI, h_Pr + proj(Dm['attn_all'].astype(np.float32), AO))
                T['M_mismatch'][0] += score_rank(v1, ds, sid_d, SUP_ID) - B['sd0']
                T['M_mismatch'][2] += 1
            nm_abs = float(np.mean([np.linalg.norm(D['head%d' % h]) for h in range(NH)]))
            for r in range(VRP):
                vr = rng.standard_normal(HID).astype(np.float32)
                vr = vr / np.linalg.norm(vr) * nm_abs
                v1 = fwd_patch(TMPL % rw, PRI, h_Pr + proj(vr, AO))
                T['V_rand'][0] += score_rank(v1, ds, sid_d, SUP_ID) - B['sd0']
                T['V_rand'][2] += 1
            n_fw += len(NAMES) + 1 + VRP
        TOUT = {}
        for k, (sd, sr, c) in T.items():
            TOUT[k] = dict(dDonor=sd / c if c else 0.0, dRecip=sr / c if c else 0.0, n=c)
        compT = {k: TOUT[k]['dDonor'] for k in DENOM}
        eff = {k: abs(compT[k]) / max(vb[k], 1e-12) for k in DENOM}
        te = sum(eff.values())
        share_eff = {k: (v / te if te > 0 else 0.0) for k, v in eff.items()}
        maxe = max(share_eff[k] for k in HEADN)
        I_nl = abs(TOUT['diff6']['dDonor']) / max(sum(abs(compT[k]) for k in DENOM), 1e-12)
        maxcomp = max(abs(v) for v in compT.values())
        floor_V = abs(TOUT['V_rand']['dDonor'])
        floor_M = abs(TOUT['M_mismatch']['dDonor'])
        floors_ok = bool(floor_V < FL['FLOOR_FRAC'] * maxcomp and floor_M < FL['FLOOR_FRAC'] * maxcomp
                         and not bad_base)
        rec['M3'] = dict(T=TOUT, comp_donor=compT, share_eff=dict(share_eff),
                         max_head_share_eff=float(maxe), I_nl=float(I_nl),
                         floors=dict(V_rand=float(floor_V), M_mismatch=float(floor_M),
                                     frac_V=float(floor_V / max(maxcomp, 1e-12)),
                                     frac_M=float(floor_M / max(maxcomp, 1e-12)), floors_ok=floors_ok))
        w('  [M3] dDonor: diff5=%+.3f attn_all=%+.3f mlp=%+.3f diff6=%+.3f I_nl=%.3f'
          % (TOUT['diff5']['dDonor'], TOUT['attn_all']['dDonor'], TOUT['mlp']['dDonor'],
             TOUT['diff6']['dDonor'], I_nl))
        w('  [M3] max_head_share_eff=%.4f ; V_rand=%+.4f M_mismatch=%+.4f floors_ok=%s'
          % (maxe, TOUT['V_rand']['dDonor'], TOUT['M_mismatch']['dDonor'], floors_ok))

    # ---------------- E9 确认集（n=17；U 仍是 discovery 估计）----------------
    conf = {}
    if CONF:
        CONFPT = [p for p in PAIRS if p[0] in CONF_W]
        CV = {k: [0.0, 0] for k in DENOM}
        for (rw, rs, dw_, ds, sw) in CONFPT:
            D = comp_deltas(rw, dw_)
            for nm in DENOM:
                pv = proj(D[nm].astype(np.float32), AO)
                CV[nm][0] += float(np.linalg.norm(pv)); CV[nm][1] += 1
        cvb = {k: (v[0] / v[1] if v[1] else 0.0) for k, v in CV.items()}
        ctv = sum(cvb.values())
        csv = {k: (v / ctv if ctv > 0 else 0.0) for k, v in cvb.items()}
        conf = dict(n=len(CONFPT), share_v_mlp=float(csv['mlp']),
                    max_head_share_v=float(max(csv[k] for k in HEADN)),
                    argmax_head_v=str(max(HEADN, key=lambda k: csv[k])), share_v=dict(csv))
        conf['G1_core'] = bool(conf['max_head_share_v'] <= FL['G1_MAXHEAD_V']
                               and conf['share_v_mlp'] <= FL['G1_MLP_SHARE_V'])
        w('  [E9] conf n=%d mlp_share_v=%.6f max_head_share_v=%.6f' %
          (conf['n'], conf['share_v_mlp'], conf['max_head_share_v']))
    rec['confirmation'] = conf

    # ---------------- 逐臂判决量 ----------------
    G1_core = bool(maxv <= FL['G1_MAXHEAD_V'] and mlp_share_v <= FL['G1_MLP_SHARE_V'])
    g1_full = G1_core
    if not NOFX and isinstance(rec.get('M3'), dict) and 'max_head_share_eff' in rec['M3']:
        g1_full = bool(G1_core and rec['M3']['max_head_share_eff'] <= FL['G1_MAXHEAD_V']
                       and rec['M3']['floors']['floors_ok'])
    rec['G1_core'] = G1_core
    rec['G1_full'] = g1_full
    rec['verdict'] = 'G1_core_distributed' if G1_core else 'G1_core_FAIL'
    rec['n_fw'] = n_fw
    rec['elapsed_s'] = round(time.time() - t0, 1)
    rec['seal_sha8'] = SEAL_SHA8
    rec['exec_sha8'] = EXEC_SHA8
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec


# ---------------------------------------------------------------- MERGE
def merge():
    recs = {}
    for a in ARM_ORDER:
        p = os.path.join(P21T, '_armrec21_%s.json' % a)
        assert os.path.exists(p), 'missing arm record %s' % p
        recs[a] = json.load(io.open(p, encoding='utf-8'))
    pairs = []
    for a_no, a_bf in (('A0_nf4', 'A0_bf16'), ('A1_nf4', 'A1_bf16')):
        Rn, Rb = recs[a_no], recs[a_bf]
        m1n, m1b = Rn['M1'], Rb['M1']
        shn = [m1n['share_v'][k] for k in (['head%d' % h for h in range(32)] + ['mlp'])]
        shb = [m1b['share_v'][k] for k in (['head%d' % h for h in range(32)] + ['mlp'])]
        rho = spearman(shn, shb)
        w_ok = None
        if 'max_head_share' in Rn.get('M2', {}) and 'max_head_share' in Rb.get('M2', {}):
            w_ok = dict(d_max=float(Rn['M2']['max_head_share'] - Rb['M2']['max_head_share']),
                        argmax_same=bool(Rn['M2']['argmax_head'] == Rb['M2']['argmax_head']))
        pairs.append(dict(
            model=Rn['model'], a_nf4=a_no, a_bf16=a_bf,
            share_v_mlp_nf4=float(m1n['share_v_mlp']), share_v_mlp_bf16=float(m1b['share_v_mlp']),
            d_share_v_mlp=float(m1n['share_v_mlp'] - m1b['share_v_mlp']),
            max_head_share_v_nf4=float(m1n['max_head_share_v']),
            max_head_share_v_bf16=float(m1b['max_head_share_v']),
            d_max_head_share_v=float(m1n['max_head_share_v'] - m1b['max_head_share_v']),
            argmax_head_v_nf4=str(m1n['argmax_head_v']), argmax_head_v_bf16=str(m1b['argmax_head_v']),
            argmax_head_v_same=bool(m1n['argmax_head_v'] == m1b['argmax_head_v']),
            spearman_share_v=rho,
            G1_core_nf4=bool(Rn['G1_core']), G1_core_bf16=bool(Rb['G1_core']),
            W=w_ok,
            W_max_nf4=(float(Rn['M2']['max_head_share']) if 'max_head_share' in Rn.get('M2', {}) else None),
            W_max_bf16=(float(Rb['M2']['max_head_share']) if 'max_head_share' in Rb.get('M2', {}) else None),
            conf_same_band=bool(Rn.get('confirmation', {}).get('G1_core') == Rb.get('confirmation', {}).get('G1_core'))
            if Rn.get('confirmation') and Rb.get('confirmation') else None,
            offload_bf16=bool(Rb.get('offload')),
        ))
    return recs, pairs


def calibrate(rec):
    """A0_bf16 对 P8 冻结锚。"""
    m1 = rec['M1']; m2 = rec.get('M2', {})
    out = dict(applies=bool(rec['scheme'] == 'bf16' and rec['model'] == 'qwen3-4b'))
    if not out['applies']:
        return out
    out['share_v_mlp'] = dict(got=float(m1['share_v_mlp']), exp=float(AP8['share_v_mlp']),
                              d=float(abs(m1['share_v_mlp'] - AP8['share_v_mlp'])))
    out['max_head_share_v'] = dict(got=float(m1['max_head_share_v']), exp=float(AP8['max_head_share_v']),
                                   d=float(abs(m1['max_head_share_v'] - AP8['max_head_share_v'])))
    out['argmax_head_v'] = dict(got=str(m1['argmax_head_v']), exp=str(AP8['argmax_head_v']),
                                same=bool(m1['argmax_head_v'] == AP8['argmax_head_v']))
    if 'max_head_share' in m2:
        out['W_max_head_share'] = dict(got=float(m2['max_head_share']), exp=float(AP8['W_max_head_share']),
                                       d=float(abs(m2['max_head_share'] - AP8['W_max_head_share'])))
        out['W_argmax_head'] = dict(got=int(m2['argmax_head']), exp=int(AP8['W_argmax_head']),
                                    same=bool(int(m2['argmax_head']) == int(AP8['W_argmax_head'])))
    if rec.get('M3') and 'I_nl' in rec.get('M3', {}):
        out['I_nl'] = dict(got=float(rec['M3']['I_nl']), exp=float(AP8['I_nl']),
                           d=float(abs(rec['M3']['I_nl'] - AP8['I_nl'])))
        out['T_diff6'] = dict(got=float(rec['M3']['T']['diff6']['dDonor']), exp=float(AP8['T_diff6_dDonor']),
                              d=float(abs(rec['M3']['T']['diff6']['dDonor'] - AP8['T_diff6_dDonor'])))
    tol = float(FL['CALIB_TOL_SHARE_V'])
    out['ok'] = bool(out['share_v_mlp']['d'] <= tol and out['max_head_share_v']['d'] <= tol
                     and out['argmax_head_v']['same']
                     and (('W_max_head_share' not in out) or out['W_max_head_share']['d'] <= float(FL['CALIB_TOL_W'])))
    return out


def predictions_check(recs, pairs):
    P = {}
    cal = calibrate(recs['A0_bf16'])
    P['P1_calib_A0bf16_reproduces_P8'] = bool(cal.get('ok'))
    ds = [p['d_share_v_mlp'] for p in pairs]
    dmx = [p['d_max_head_share_v'] for p in pairs]
    P['P2_share_v_mlp_stable'] = bool(all(abs(x) <= FL['QUANT_TOL_SHARE_V'] for x in ds))
    P['P3_max_head_share_v_stable'] = bool(all(abs(x) <= FL['QUANT_TOL_MAXHEAD_V'] for x in dmx))
    P['P4_argmax_head_v_same'] = bool(all(p['argmax_head_v_same'] for p in pairs))
    P['P5_G1_core_both_precisions'] = bool(all(p['G1_core_nf4'] and p['G1_core_bf16'] for p in pairs))
    P['P6_W_stable'] = bool(all((p['W'] is not None) and abs(p['W']['d_max']) <= FL['QUANT_TOL_W']
                                and p['W']['argmax_same'] for p in pairs))
    P['P7_spearman_share_v'] = bool(all((p['spearman_share_v'] is not None)
                                        and p['spearman_share_v'] >= FL['SPEARMAN_MIN'] for p in pairs))
    P['P8_conf_same_band'] = bool(all(p['conf_same_band'] in (True, None) for p in pairs))
    fl_ok = []
    for a in ARM_ORDER:
        m3 = recs[a].get('M3') or {}
        if m3 and 'floors' in m3:
            fl_ok.append(bool(m3['floors']['floors_ok']))
    P['P9_floors'] = bool(all(fl_ok)) if fl_ok else None
    return P


def main():
    if MERGE:
        recs, pairs = merge()
        cal = calibrate(recs['A0_bf16'])
        P = predictions_check(recs, pairs)
        out = dict(phase=21, line='N2h1-alpha-14', kind='result',
                   smoke=SMOKE, nofx=NOFX,
                   seal_sha8=SEAL_SHA8, exec_sha8=EXEC_SHA8,
                   arm_order=ARM_ORDER,
                   arms={a: recs[a] for a in ARM_ORDER},
                   calibration=cal, quant_pairs=pairs, predictions=P,
                   n_pass=int(sum(1 for v in P.values() if v is True)),
                   n_total=int(sum(1 for v in P.values() if v is not None)),
                   p8_anchor=AP8,
                   note='M1（share_v，向量预算/精确可加）为第一指标；M2=权重容量；M3=效应侧（第二指标）。')
        with io.open(RESULT, 'w', encoding='utf-8', newline='\n') as f:
            f.write(json.dumps(out, ensure_ascii=False, indent=1))
        w('=== MERGE DONE -> %s ===' % RESULT)
        for k, v in P.items():
            w('  %-38s %s' % (k, v))
        w('  calib ok=%s' % cal.get('ok'))
        for p in pairs:
            w('  %s d_mlp=%+.6f d_maxv=%+.6f argmax_same=%s rho=%s G1(nf4/bf16)=%s/%s'
              % (p['model'], p['d_share_v_mlp'], p['d_max_head_share_v'], p['argmax_head_v_same'],
                 F3(p['spearman_share_v'], 4), p['G1_core_nf4'], p['G1_core_bf16']))
        return

    todo = [a for a in ARM_ORDER if (not ARMS_SEL) or a == ARMS_SEL]
    assert todo, 'no arm selected (ARMS=%r)' % ARMS_SEL
    for a in todo:
        w('')
        w('================ ARM %s ================' % a)
        rec = run_arm(a, ARMS_CFG[a])
        sub = 'smoke' if SMOKE else ''
        d = os.path.join(P21T, sub) if sub else P21T
        fp = os.path.join(d, '_armrec21_%s%s.json' % (a, '_SMOKE' if SMOKE else ''))
        with io.open(fp, 'w', encoding='utf-8', newline='\n') as f:
            f.write(json.dumps(rec, ensure_ascii=False, indent=1))
        w('  arm %s -> %s (%.1fs, n_fw=%d)' % (a, fp, rec['elapsed_s'], rec['n_fw']))
        del rec
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    print('MAIN DONE')
