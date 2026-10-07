# -*- coding: utf-8 -*-
"""
Phase 15 (N2h1-alpha-8) 主脚本：跨模型复算「统一剖面」。
三臂（同一 nf4 口径，串行）：
  A0_calib_qwen3-4b-nf4 : 量化保真校准（与 Phase 12 bf16 已发表量逐位点比对）
  A1_glm4-9b-nf4        : 跨家族 untied 复算
  A2_qwen3-14b-nf4      : 同家族 untied 规模放大复算
每臂流程：E0 装置自检 -> E1 capture(41) -> E2 FULL_SWAP -> E3 独立写入窗定位 -> E4 单点替换族双坐标剖面
          -> E5 集中度 + 置换零假设 -> (A0) E6 量化保真校准
判据 Q0-Q5 全部预先冻结在 seal 中；本脚本只执行不做判断。
用法：SMOKE=1 python n2h1a8_cross_model_profile.py    (仅 A0，缩小网格)
      python n2h1a8_cross_model_profile.py            (三臂正式)
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
P15 = os.path.join(ROOT, 'tests', 'deepseek', 'Phase15')
P15T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase15')
EXECP = os.path.join(P15T, 'execution_phase15.json')
SEALP = os.path.join(P15T, 'N2h1a8_design_seal.json')
SMOKE = os.environ.get('SMOKE', '0') == '1'
ARMS_SEL = os.environ.get('ARMS', '')
# 进程隔离模式：同一进程内连续加载 3 个 nf4 大模型会段错误（EXIT=139，
# 实测在第 3 个模型加载到 ~82% 处崩），故支持「每臂一进程 + 合并」。
SPLIT_PARTIAL = os.environ.get('SPLIT_PARTIAL', '0') == '1'
MERGE = os.environ.get('MERGE', '0') == '1'


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


EX = json.load(io.open(EXECP, encoding='utf-8'))
assert sha(SEALP) == EX['seal_sha256'], 'DRIFT: seal sha != execution.seal_sha256'

# amend1（词表常量勘误）：sup_id 必须逐臂由该臂 tokenizer 解析，禁止沿用源模型硬编码 id
AM1P = os.path.join(P15T, 'N2h1a8_design_seal_amend1.json')
AM1 = json.load(io.open(AM1P, encoding='utf-8'))
assert AM1['amend_of_seal_sha256'] == EX['seal_sha256'], \
    'DRIFT: amend1 指向的 seal 与本 exec 不一致'

TMPL = EX['template']
SUP_ID_REF = {k: int(v) for k, v in EX['sup_id'].items()}   # qwen 族参考值（仅用于自洽核对）
SUPS = list(EX['classes'])
DISC = [tuple(x) for x in EX['discovery']]
CONF = [tuple(x) for x in EX['confirmation']]
INST_ALL = [tuple(x) for x in EX['instances_all']]
PAIRS_ALL = [tuple(x) for x in EX['pairs_all']]
PROFILE = list(EX['profile_sites'])
ALPHAS = list(EX['alphas'])
CANDS = list(EX['localize']['cands'])
W = int(EX['W'])
XHF = float(EX['xh_frac'])
BP = int(EX['bootstrap']['BP'])
SEED = int(EX['bootstrap']['seed'])
FL = EX['floors']
INH = EX['inheritance']
QUANT = EX['quant']

if SMOKE:
    ALPHAS = [0.0, 0.5, 1.0]
    PROFILE = PROFILE[:3]
    CANDS = [4, 6, 20]
    BP = 200

_log = []


def w(s=''):
    _log.append(str(s))
    print(s)
    sys.stdout.flush()


def flush_log(name):
    io.open(os.path.join(P15T, name), 'w', encoding='utf-8', newline='\n').write('\n'.join(_log) + '\n')


# ---------------------------------------------------------------- 通用工具
def attn_out_proj(layer):
    a = layer.self_attn
    for nm in ('o_proj', 'dense', 'out_proj'):
        if hasattr(a, nm):
            return getattr(a, nm), nm
    raise RuntimeError('no attn out proj')


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 1e-12 else None


def cross_alpha(xs, ys, frac):
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
    """峰值斜率 / 其余斜率中位数（逐字沿用 Phase 12）。"""
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


def conc_hat(F, wwin):
    """top3_share = max_j |sum(jm[j:j+W])| / range（逐字沿用 Phase 12/13/14）。"""
    F = np.asarray(F, float)
    jm = np.diff(F)
    if not np.isfinite(F).all():
        return None, None, [float(v) for v in jm]
    rng = float(F.max() - F.min())
    if rng <= 1e-12 or len(jm) < wwin:
        return None, None, jm.tolist()
    wins = [abs(float(np.sum(jm[j:j + wwin]))) for j in range(len(jm) - wwin + 1)]
    k = int(np.argmax(wins))
    return float(wins[k] / rng), k, jm.tolist()


def perm_null(jumps, rng_obj, wwin, n_bp, range_val):
    js = np.asarray(jumps, float)
    # 边界：jumps 数 < 窗口宽（SMOKE 网格下会发生）或含 nan -> 不产出 null，显式记录原因
    if len(js) < wwin:
        return dict(BP=n_bp, null95=None, ci=None, n_ok=0,
                    reason='n_jumps(%d) < W(%d)' % (len(js), wwin))
    if not np.isfinite(js).all():
        return dict(BP=n_bp, null95=None, ci=None, n_ok=0, reason='non_finite_jumps')
    out = np.full(n_bp, np.nan)
    for b in range(n_bp):
        p = js[rng_obj.permutation(len(js))]
        wins = [abs(float(np.sum(p[j:j + wwin]))) for j in range(len(p) - wwin + 1)]
        out[b] = max(wins) / max(range_val, 1e-12)
    fin = out[np.isfinite(out)]
    if len(fin) == 0:
        return dict(BP=n_bp, null95=None, ci=None, n_ok=0, reason='all_nan')
    return dict(BP=n_bp, null95=float(np.percentile(fin, 95)),
                ci=dict(lo=float(np.percentile(fin, 2.5)), hi=float(np.percentile(fin, 97.5)),
                        med=float(np.median(fin))),
                n_ok=int(len(fin)))


# ---------------------------------------------------------------- 单臂
def run_arm(arm_id, acfg):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    rec = dict(arm=arm_id, role=acfg['role'], model=acfg['model'], smoke=SMOKE)
    w('')
    w('=' * 78)
    w('ARM %s  (%s ; %s)' % (arm_id, acfg['model'], acfg['role']))
    t_arm = time.time()
    MDIR = os.path.join(ROOT, 'models', 'hf', acfg['dir'])

    # --- F0 config 冻结校验
    csha = sha(os.path.join(MDIR, 'config.json'))
    rec['config_sha256'] = csha
    rec['config_sha_match'] = (csha == acfg['config_sha256'])
    w('  F0 config sha8 = %s ; 与 seal 一致 = %s' % (csha[:8], rec['config_sha_match']))
    assert rec['config_sha_match'], 'F0 失败：config 漂移'

    tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)

    def ids_of(s):
        return tok.encode(s, add_special_tokens=False)

    # --- F1b 类别 token 逐臂解析（amend1）
    # 事故：seal 的 sup_id 是 qwen 词表 id，被全局用于三臂；glm4-9b 词表不同
    # （vocab 151329 vs 151643）⇒ A1 全程读错类别 token（F2_base_ok 当场抓到 bad=23/41）。
    # 修正：每臂用自己的 tokenizer 现场解析，并要求 6/6 类别词恰为单 token 且 decode 可逆。
    SUP_ID = {}
    for _wd in SUPS:
        _t = list(tok.encode(_wd, add_special_tokens=False))
        assert len(_t) == 1 and tok.decode([_t[0]]) == _wd, \
            'F1b 失败：类别词 %r 非单 token 或 decode 不可逆 %r' % (_wd, _t)
        SUP_ID[_wd] = int(_t[0])
    rec['sup_id_arm'] = dict(SUP_ID)
    rec['sup_id_matches_ref'] = bool(all(SUP_ID[k] == SUP_ID_REF[k] for k in SUPS))
    rec['F1b_ok'] = True
    w('  F1b 类别 token（逐臂解析）: %s ; 与 qwen 参考值一致=%s'
      % (json.dumps(SUP_ID, ensure_ascii=False), rec['sup_id_matches_ref']))

    # --- E1a T=2 布局核验
    tl = {}
    for wd, sup in INST_ALL:
        tl.setdefault(len(ids_of(TMPL % wd)), []).append(wd)
    rec['token_len_hist'] = {str(k): len(v) for k, v in sorted(tl.items())}
    t2_only = (sorted(tl.keys()) == [2])
    rec['T2_only'] = bool(t2_only)
    w('  F1 T=2 布局: hist=%s ; only_T2=%s' % (rec['token_len_hist'], t2_only))
    if not t2_only:
        rec['non_T2'] = {str(k): v for k, v in sorted(tl.items()) if k != 2}

    # --- 加载（统一 nf4 口径）
    # 注意：max_memory 经 JSON 往返后键为字符串，accelerate 要求 GPU 用整数键 -> 运行期转换
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
    HID = model.config.hidden_size
    CFG = model.config
    NH = int(CFG.num_attention_heads)
    HDP = int(getattr(CFG, 'head_dim', HID // max(NH, 1)))
    KVH = int(getattr(CFG, 'num_key_value_heads', -1))
    oproj, onm = attn_out_proj(layers[0])
    OIN = int(oproj.in_features)
    dev_hist = {}
    for nm_, p in model.named_parameters():
        dev_hist[str(p.device)] = dev_hist.get(str(p.device), 0) + 1
    rec['cfg'] = dict(L=L, hid=HID, n_heads=NH, kv_heads=KVH, head_dim=HDP, o_proj_in=OIN,
                      o_proj_name=onm, tie=bool(getattr(CFG, 'tie_word_embeddings', False)),
                      vocab=int(CFG.vocab_size), param_devices=dev_hist)
    exp = acfg['expected']
    drift = []
    if L != exp['num_hidden_layers']:
        drift.append('L')
    if HID != exp['hidden_size']:
        drift.append('hid')
    if NH != exp['num_attention_heads']:
        drift.append('n_heads')
    if KVH != exp['num_key_value_heads']:
        drift.append('kv_heads')
    if bool(getattr(CFG, 'tie_word_embeddings', False)) != bool(exp['tie_word_embeddings']):
        drift.append('tie')
    if OIN != NH * HDP:
        drift.append('o_proj_in')
    if max(PROFILE) >= L - 1 or max(CANDS) >= L - 1:
        CANDS2 = [c for c in CANDS if c < L - 1]
    else:
        CANDS2 = list(CANDS)
    rec['drift'] = drift
    rec['F4_dims_ok'] = (len(drift) == 0)
    rec['F5_o_proj_ok'] = (OIN == NH * HDP)
    w('  F4 维度: L=%d hid=%d heads=%d kv=%d head_dim=%d o_proj_in=%d(%s) tie=%s ; drift=%s' %
      (L, HID, NH, KVH, HDP, OIN, onm, rec['cfg']['tie'], drift if drift else 'NONE'))
    w('  param devices: %s' % dev_hist)
    rec['cands_used'] = CANDS2
    assert max(PROFILE) < L - 1, 'profile_sites 超出层数'

    # --- 前向原语
    @torch.no_grad()
    def fwd_patch(text, site, vec):
        ii = torch.tensor([ids_of(text)], device='cuda')
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

    @torch.no_grad()
    def capture(text):
        ii = torch.tensor([ids_of(text)], device='cuda')
        o = model(input_ids=ii, output_hidden_states=True)
        HH = np.stack([x[0, -1].float().detach().cpu().numpy() for x in o.hidden_states], 0)
        return HH, o.logits[0, -1].float().detach().cpu().numpy()

    def score_of(v, sup, sid):
        v = v.copy(); v[sid] = -1e9
        own = float(v[SUP_ID[sup]])
        others = [float(v[SUP_ID[x]]) for x in SUPS if x != sup]
        return own - float(np.mean(others))

    def rank_of(v, sup, sid):
        v = v.copy(); v[sid] = -1e9
        return int(np.where(np.argsort(-v) == SUP_ID[sup])[0][0]) + 1

    # --- E0 装置自检
    ii0 = torch.tensor([ids_of(TMPL % '苹果')], device='cuda')
    with torch.no_grad():
        lga = model(input_ids=ii0).logits[0, -1].float().detach().cpu().numpy()
        lgb = model(input_ids=ii0).logits[0, -1].float().detach().cpu().numpy()
    determinism = float(np.max(np.abs(lga - lgb)))
    rng0 = np.random.default_rng(SEED)
    vec0 = (rng0.standard_normal(HID) * 0.1).astype(np.float32)
    lg_hook = fwd_patch(TMPL % '苹果', L // 2, torch.tensor(vec0, device='cuda'))
    hook_effect = float(np.max(np.abs(lg_hook - lgb)))
    rec['E0_selfcheck'] = dict(determinism_maxdiff=determinism, hook_site=L // 2,
                              hook_effect_maxdiff=hook_effect, o_proj_ok=rec['F5_o_proj_ok'],
                              T2_only=bool(t2_only), dims_ok=rec['F4_dims_ok'])
    w('  E0 自检: determinism=%.3e ; hook@L%d effect=%.3e' % (determinism, L // 2, hook_effect))

    # --- E1 capture
    t0 = time.time()
    CAP = {}
    for wd, sup in INST_ALL:
        CAP[wd] = capture(TMPL % wd)
    rec['E1_capture'] = dict(n=len(CAP), seconds=round(time.time() - t0, 1),
                            hidden_levels=int(CAP[INST_ALL[0][0]][0].shape[0]))
    w('  E1 capture %d 实例 / %.1fs (hidden levels=%d)' %
      (len(CAP), rec['E1_capture']['seconds'], rec['E1_capture']['hidden_levels']))
    if SMOKE:
        HH0, lg0 = CAP[INST_ALL[0][0]]
        assert HH0.shape[1] == HID and not np.isnan(HH0).any()
        w('  SMOKE capture assert: HH%s logits%s' % (HH0.shape, lg0.shape))
    PAIRS = [p for p in PAIRS_ALL if p[0] in CAP and p[2] in CAP]
    H1 = 1  # hidden_states[i] = layers[i-1] 输出；HH[s+1] = layers[s] 输出

    # --- E1b BASE
    BASE = {}
    for (rw, rs, dw, ds, sw) in PAIRS:
        lg = CAP[rw][1]
        BASE[rw] = dict(sr0=score_of(lg, rs, ids_of(rw)[0]),
                        sd0=score_of(lg, ds, ids_of(dw)[0]),
                        rd0=rank_of(lg, ds, ids_of(dw)[0]))
    bad_base = [rw for rw, b in BASE.items() if not (b['sr0'] > 0)]
    w('  F2 base: n=%d ; 受体类分数 mean=%+.3f ; 供体类分数 mean=%+.3f ; bad=%s' %
      (len(BASE), float(np.mean([b['sr0'] for b in BASE.values()])),
       float(np.mean([b['sd0'] for b in BASE.values()])), bad_base if bad_base else 'NONE'))
    rec['F2_base_ok'] = (not bad_base)
    rec['F2_base_bad'] = bad_base

    # --- E2 FULL_SWAP（零额外前向）
    FS_PAIR = {}
    for (rw, rs, dw, ds, sw) in PAIRS:
        FS_PAIR[rw] = float(score_of(CAP[dw][1], ds, ids_of(dw)[0]) - BASE[rw]['sd0'])
    DISC_P = [p for p in PAIRS if p[0] in [x[0] for x in DISC]]
    CONF_P = [p for p in PAIRS if p[0] in [x[0] for x in CONF]]
    FULL_SWAP = float(np.mean([FS_PAIR[p[0]] for p in DISC_P]))
    FS_ORDER = [p[0] for p in DISC_P]
    FS_VEC = np.array([FS_PAIR[rw] for rw in FS_ORDER], float)
    rec['E2_full_swap'] = dict(FULL_SWAP=FULL_SWAP, n=len(FS_VEC),
                               min=float(FS_VEC.min()), max=float(FS_VEC.max()),
                               mean_rank1=float(np.mean([1.0 if BASE[rw]['rd0'] == 1 else 0.0
                                                         for rw in FS_ORDER])),
                               FS_ORDER=list(FS_ORDER),
                               FS_VEC=[float(x) for x in FS_VEC],
                               FS_PAIR={str(k): float(v) for k, v in FS_PAIR.items()},
                               DISC_PAIRS=[[str(x) for x in p] for p in DISC_P],
                               BASE_sr0={str(rw): float(BASE[rw]['sr0']) for rw in FS_ORDER},
                               BASE_sd0={str(rw): float(BASE[rw]['sd0']) for rw in FS_ORDER})
    w('  E2 FULL_SWAP = %+.6f (n=%d ; range %+.3f..%+.3f ; 供体类已 rank1 比例 %.3f)' %
      (FULL_SWAP, len(FS_VEC), FS_VEC.min(), FS_VEC.max(), rec['E2_full_swap']['mean_rank1']))

    # --- VEC（每对：各 profile 位点的受体/供体状态与差向量）
    VEC = {}
    for (rw, rs, dw, ds, sw) in PAIRS:
        h_ell, hd_ell, nh_ell, d_ell = {}, {}, {}, {}
        for s in PROFILE:
            hr = CAP[rw][0][s + 1].astype(np.float32)
            hd = CAP[dw][0][s + 1].astype(np.float32)
            h_ell[s] = hr
            hd_ell[s] = hd
            nh_ell[s] = float(np.linalg.norm(hr))
            d_ell[s] = hd - hr
        VEC[rw] = dict(h_ell=h_ell, hd_ell=hd_ell, nh_ell=nh_ell, d_ell=d_ell)
    Q_ELL = {str(s): float(np.mean([np.linalg.norm(VEC[rw]['d_ell'][s]) /
                                    max(VEC[rw]['nh_ell'][s], 1e-9) for rw in VEC]))
             for s in PROFILE}
    w('  q_ell (= a_rel_full): %s' %
      '  '.join('L%d:%.3f' % (s, Q_ELL[str(s)]) for s in PROFILE))
    rec['q_ell'] = Q_ELL

    # --- F3 alpha=0 还原基线
    f3 = {}
    for site in sorted(set([PROFILE[0], PROFILE[len(PROFILE) // 2], PROFILE[-1]])):
        devs = []
        for (rw, rs, dw, ds, sw) in DISC_P[:3]:
            h0 = VEC[rw]['h_ell'][site]
            lg0 = fwd_patch(TMPL % rw, site, torch.tensor(h0, device='cuda'))
            devs.append(abs(score_of(lg0, rs, ids_of(rw)[0]) - BASE[rw]['sr0']))
        f3[str(site)] = float(max(devs))
    rec['F3_alpha0_maxdev'] = f3
    w('  F3 alpha=0 还原: %s' % '  '.join('%s=%.3e' % (k, v) for k, v in f3.items()))

    # --- E4 单点替换族双坐标剖面
    def dose_swap(site):
        out = []
        for a in ALPHAS:
            dd = 0.0; per = []; order = []; perts = []; n = 0
            for (rw, rs, dw, ds, sw) in DISC_P:
                V, B = VEC[rw], BASE[rw]
                h0 = V['h_ell'][site]
                dv = V['d_ell'][site]
                lg = fwd_patch(TMPL % rw, site, torch.tensor(h0 + a * dv, device='cuda'))
                x1 = score_of(lg, ds, ids_of(dw)[0]) - B['sd0']
                dd += x1; n += 1
                per.append(float(x1)); order.append(rw)
                perts.append(a * float(np.linalg.norm(dv)) / max(V['nh_ell'][site], 1e-9))
            out.append(dict(alpha=float(a), dDonor=dd / max(n, 1), n=n,
                            pert_rel=float(np.mean(perts)), per_pair=per, order=order))
        return out

    t0 = time.time()
    E4 = {}
    for s in PROFILE:
        E4[str(s)] = dose_swap(s)
        w('    L%-3d %s' % (s, ' '.join('a=%.2f dD=%+7.3f' % (r['alpha'], r['dDonor']) for r in E4[str(s)])))
    rec['E4_seconds'] = round(time.time() - t0, 1)
    rec['E4_profile'] = E4
    w('  E4 剖面完成 %d 位点 x %d alpha x %d pairs / %.1fs' %
      (len(PROFILE), len(ALPHAS), len(DISC_P), rec['E4_seconds']))

    # 逐对均值曲线 + 双坐标
    PM = np.stack([np.array([r['per_pair'] for r in E4[str(s)]], float) for s in PROFILE], 0)  # [nS, nA, nP]
    Y = PM.mean(axis=2) / FULL_SWAP
    xh = np.array([cross_alpha(ALPHAS, Y[i], XHF) for i in range(len(PROFILE))],
                  dtype=float)
    xh = np.where(np.isfinite(xh), xh, np.nan)
    Jv = np.array([J_only(ALPHAS, Y[i]) for i in range(len(PROFILE))], dtype=float)
    xh_sites = [PROFILE[i] for i in range(len(PROFILE)) if np.isfinite(xh[i])]
    rec['E4_summary'] = dict(sites=PROFILE, alphas=ALPHAS,
                             xhalf=[float(v) for v in xh], J=[float(v) for v in Jv],
                             n_finite_x=int(np.isfinite(xh).sum()),
                             n_finite_j=int(np.isfinite(Jv).sum()),
                             XH_RANGE=(float(np.nanmax(xh) - np.nanmin(xh)) if np.isfinite(xh).any() else None),
                             recover=[float(Y[i, list(ALPHAS).index(1.0)] if 1.0 in ALPHAS else np.nan)
                                      for i in range(len(PROFILE))])
    w('  E4 双坐标:')
    for i, s in enumerate(PROFILE):
        w('    L%-3d xhalf=%s J=%s' %
          (s, ('%.6f' % xh[i]) if np.isfinite(xh[i]) else 'None',
           ('%.4f' % Jv[i]) if np.isfinite(Jv[i]) else 'None'))

    # --- E3 独立写入窗定位（逐层独立 U；B_cat 口径）
    def est_U(level_idx):
        by = {}
        for wd, sup in INST_ALL:
            if wd in CAP:
                by.setdefault(sup, []).append(wd)
        avail = [s for s in SUPS if s in by]
        r = max(len(avail) - 1, 1)
        mus = np.stack([np.mean([CAP[wd][0][level_idx] for wd in by[s]], 0) for s in avail], 0).astype(np.float64)
        D = mus - mus.mean(0, keepdims=True)
        _, sv, Vt = np.linalg.svd(D, full_matrices=False)
        return Vt[:r].astype(np.float32), sv[:r], len(avail)

    def proj(vec, Ub):
        return (vec @ Ub.T) @ Ub

    t0 = time.time()
    curve = {}
    loc_meta = {}
    for l in CANDS2:
        Ub, sv, ncls = est_U(l + 1)
        loc_meta[str(l)] = dict(sing=[float(x) for x in sv], n_classes=int(ncls), rank=int(Ub.shape[0]))
        acc = 0.0; n = 0
        for (rw, rs, dw, ds, sw) in DISC_P:
            hr = CAP[rw][0][l + 1].astype(np.float32)
            hd = CAP[dw][0][l + 1].astype(np.float32)
            vec = hr + proj((hd - hr).astype(np.float32), Ub)
            lg = fwd_patch(TMPL % rw, l, torch.tensor(vec, device='cuda'))
            acc += score_of(lg, ds, ids_of(dw)[0]) - BASE[rw]['sd0']
            n += 1
        curve[l] = acc / max(n, 1)
        w('    L%-3d B_cat=%+8.3f' % (l, curve[l]))
    PL = [l for l in CANDS2 if l in curve]
    jumps = [(PL[i], curve[PL[i]] - curve[PL[i - 1]]) for i in range(1, len(PL))]
    if jumps:
        lstar, ljv = max(jumps, key=lambda x: x[1])
    else:
        lstar, ljv = None, None
    rec['E3_localize'] = dict(cands=PL, curve={str(k): float(v) for k, v in curve.items()},
                              adjacent_jumps=[(int(a), float(b)) for a, b in jumps],
                              L_star_own=(int(lstar) if lstar is not None else None),
                              L_star_increment=(float(ljv) if ljv is not None else None),
                              seconds=round(time.time() - t0, 1), U_meta=loc_meta)
    w('  E3 独立定位 L*_own = %s (相邻最大增量 %+.3f) / %.1fs' %
      (lstar, (ljv if ljv is not None else float('nan')), rec['E3_localize']['seconds']))

    # --- E5 集中度 + 置换零假设（零额外前向）
    share_x, axw_x, jx = conc_hat(xh, W)
    share_j, axw_j, jj = conc_hat(Jv, W)
    rng_n = np.random.default_rng(SEED + 13)
    rng_n2 = np.random.default_rng(SEED + 29)
    range_x = float(np.nanmax(xh) - np.nanmin(xh)) if np.isfinite(xh).any() else float('nan')
    range_j = float(np.nanmax(Jv) - np.nanmin(Jv)) if np.isfinite(Jv).any() else float('nan')
    nx = perm_null(jx, rng_n, W, BP, range_x) if np.isfinite(jx).all() and np.isfinite(range_x) else dict(BP=BP, null95=None)
    nj = perm_null(jj, rng_n2, W, BP, range_j) if np.isfinite(jj).all() and np.isfinite(range_j) else dict(BP=BP, null95=None)
    margin_x = (share_x - nx['null95']) if (share_x is not None and nx.get('null95') is not None) else None
    margin_j = (share_j - nj['null95']) if (share_j is not None and nj.get('null95') is not None) else None
    rec['E5_concentration'] = dict(
        W=W, sites=PROFILE, alpha_grid=ALPHAS,
        xhalf=[float(v) for v in xh], J=[float(v) for v in Jv],
        top3_x=share_x, argmax_w_x=axw_x, jumps_x=jx,
        top3_j=share_j, argmax_w_j=axw_j, jumps_j=jj,
        range_x=range_x, range_j=range_j,
        win_sem_x=(None if axw_x is None else dict(w=axw_x, a=PROFILE[axw_x], b=PROFILE[min(axw_x + W, len(PROFILE) - 1)])),
        win_sem_j=(None if axw_j is None else dict(w=axw_j, a=PROFILE[axw_j], b=PROFILE[min(axw_j + W, len(PROFILE) - 1)])),
        null_x=nx, null_j=nj, margin_x=margin_x, margin_j=margin_j)
    w('  E5 集中度: share_x=%s (argmax_w=%s) ; share_j=%s (argmax_w=%s) ; W=%d' %
      (None if share_x is None else '%.4f' % share_x, axw_x,
       None if share_j is None else '%.4f' % share_j, axw_j, W))
    w('     null95_x=%s margin_x=%s ; null95_j=%s margin_j=%s (BP=%d)' %
      (nx.get('null95'), None if margin_x is None else '%.4f' % margin_x,
       nj.get('null95'), None if margin_j is None else '%.4f' % margin_j, BP))
    rec['E5_concentration']['spearman_xh_depth'] = spearman(
        [xh[i] for i in range(len(PROFILE)) if np.isfinite(xh[i])],
        [PROFILE[i] for i in range(len(PROFILE)) if np.isfinite(xh[i])])
    rec['E5_concentration']['spearman_J_depth'] = spearman(
        [Jv[i] for i in range(len(PROFILE)) if np.isfinite(Jv[i])],
        [PROFILE[i] for i in range(len(PROFILE)) if np.isfinite(Jv[i])])
    rec['E5_concentration']['d_argmax_window'] = (
        abs(int(axw_x) - int(axw_j)) if (axw_x is not None and axw_j is not None) else None)
    w('     spearman(xhalf,depth)=%s ; spearman(J,depth)=%s ; d_argmax_window=%s' %
      (rec['E5_concentration']['spearman_xh_depth'], rec['E5_concentration']['spearman_J_depth'],
       rec['E5_concentration']['d_argmax_window']))

    # α=1 的 y 值（recover）：单点替换族下 α=1 表示「该位点残差换成供体贴」，其下游计算与
    # 「供体自己的前向」不同，故 recover ≈ 1 但 ≠ 1（Phase 12 既有口径 recover=0.9955..1.0010）。
    # 这里不做恒等式断言，只记录数值，供 A0 与 Phase 12 bf16 已发表 recover 比对。
    a1_idx = list(ALPHAS).index(1.0) if 1.0 in ALPHAS else None
    if a1_idx is not None:
        r1 = np.array(PM[:, a1_idx, :], float) / FULL_SWAP
        rec['recover_at_alpha1'] = [float(v) for v in r1.mean(axis=1)]
        rec['recover_dev_vs_1'] = float(np.max(np.abs(r1.mean(axis=1) - 1.0)))
        w('  recover(alpha=1): %s ; max|recover-1| = %.4f' %
          (' '.join('L%d:%.4f' % (PROFILE[i], r1.mean(axis=1)[i]) for i in range(len(PROFILE))),
           rec['recover_dev_vs_1']))

    # --- E6 校准（仅 A0）
    if arm_id.startswith('A0'):
        XH12 = {int(k): float(v) for k, v in INH['XH_12_by_site'].items()}
        J12 = {int(k): float(v) for k, v in INH['J_swap_12_by_site'].items()}
        rows = []
        for i, s in enumerate(PROFILE):
            if s in XH12:
                rows.append(dict(site=int(s), xh_nf4=float(xh[i]), xh_bf16=float(XH12[s]),
                                 dxh=(float(xh[i]) - float(XH12[s])) if np.isfinite(xh[i]) else None,
                                 J_nf4=float(Jv[i]) if np.isfinite(Jv[i]) else None,
                                 J_bf16=float(J12.get(s, float('nan')))))
        dxh_vals = [abs(r['dxh']) for r in rows if r['dxh'] is not None]
        jrat = [abs(r['J_nf4'] / r['J_bf16']) for r in rows
                if r['J_nf4'] is not None and np.isfinite(r['J_bf16']) and abs(r['J_bf16']) > 1e-9]
        share_x_12, axw_x_12v, _ = conc_hat(np.array([XH12[s] for s in PROFILE], float), W)
        REC12 = {int(k): float(v) for k, v in INH['recover_12_by_site'].items()}
        drec = [abs(rec.get('recover_at_alpha1', [np.nan] * len(PROFILE))[i] - REC12[s])
                for i, s in enumerate(PROFILE) if s in REC12 and 'recover_at_alpha1' in rec]
        rec['E6_calibration'] = dict(
            rows=rows,
            max_abs_dxh=(max(dxh_vals) if dxh_vals else None),
            argmax_w_x_nf4=axw_x, argmax_w_x_bf16=axw_x_12v,
            share_x_nf4=share_x, share_x_bf16=float(INH['SHARE_X_13']),
            max_abs_drecover=(max(drec) if drec else None),
            J_ratio_min=(min(jrat) if jrat else None), J_ratio_max=(max(jrat) if jrat else None),
            XH_RANGE_nf4=rec['E4_summary']['XH_RANGE'], XH_RANGE_bf16=float(INH['XH_RANGE_12']))
        rec['E6_calibration']['pass_tol'] = (rec['E6_calibration']['max_abs_dxh'] is not None
                                             and rec['E6_calibration']['max_abs_dxh'] <= FL['XH_FAITHFUL_TOL'])
        # None==None 不算一致（SMOKE 粗网格会两者皆 None -> 必须判 False）
        rec['E6_calibration']['argmax_same'] = bool(axw_x is not None and axw_x == axw_x_12v)
        w('  E6 校准(vs Phase12 bf16): max|dxhalf|=%.4f (tol %.2f -> %s) ; argmax_w_x nf4=%s bf16=%s (same=%s)' %
          (rec['E6_calibration']['max_abs_dxh'] or float('nan'), FL['XH_FAITHFUL_TOL'],
           rec['E6_calibration']['pass_tol'], axw_x, axw_x_12v, rec['E6_calibration']['argmax_same']))
        w('     share_x nf4=%.4f vs bf16=%.4f ; max|drecover|=%s ; J 比值范围 [%s, %s] ; XH_RANGE nf4=%.4f bf16=%.4f' %
          (share_x if share_x else float('nan'), rec['E6_calibration']['share_x_bf16'],
           rec['E6_calibration']['max_abs_drecover'],
           ('%.3f' % rec['E6_calibration']['J_ratio_min']) if jrat else 'NA',
           ('%.3f' % rec['E6_calibration']['J_ratio_max']) if jrat else 'NA',
           rec['E4_summary']['XH_RANGE'] or float('nan'), rec['E6_calibration']['XH_RANGE_bf16']))
        if SMOKE:
            rec['E6_smoke_note'] = 'SMOKE 只跑前 %d 位点，校准量不完整' % len(PROFILE)

    # --- 释放
    rec['elapsed_s'] = round(time.time() - t_arm, 1)
    del model
    try:
        del CAP, VEC
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    w('  ARM %s done %.1fs' % (arm_id, rec['elapsed_s']))
    flush_log('_arm_%s.log' % arm_id)
    if SPLIT_PARTIAL:
        pp = os.path.join(P15T, '_armrec_%s.json' % arm_id)
        io.open(pp, 'w', encoding='utf-8', newline='\n').write(json.dumps(rec, ensure_ascii=False))
        w('  SPLIT_PARTIAL -> %s (%d B)' % (os.path.basename(pp), os.path.getsize(pp)))
    return rec


# ---------------------------------------------------------------- 判决
def per_arm_verdict(rec):
    c = rec.get('E5_concentration', {})
    e6 = rec.get('E6_calibration')
    v = {}
    v['Q0_device'] = ('PASS' if (rec.get('F1b_ok') and rec.get('T2_only') and rec.get('F4_dims_ok') and
                                 rec.get('F2_base_ok') and
                                 rec['E0_selfcheck']['determinism_maxdiff'] == 0 and
                                 rec['E0_selfcheck']['hook_effect_maxdiff'] > 0) else 'FAIL')
    d = c.get('d_argmax_window')
    v['Q2_d_argmax'] = d
    v['Q2_label'] = ('NA' if d is None else ('ARGS_GAP_GE3' if d >= FL['ARGS_GAP_MIN'] else 'ARGS_GAP_LT3'))
    n95x = (c.get('null_x') or {}).get('null95')
    n95j = (c.get('null_j') or {}).get('null95')
    v['Q3_null95_x'] = n95x
    v['Q3_null95_j'] = n95j
    v['Q3_label'] = ('NA' if n95x is None else ('NULL_X_HIGH' if n95x >= FL['NULL_HIGH'] else 'NULL_X_OK'))
    v['Q4_margin_x'] = c.get('margin_x')
    v['Q4_margin_j'] = c.get('margin_j')
    v['Q4_label_x'] = ('NA' if c.get('margin_x') is None else
                       ('ABOVE_NULL' if c['margin_x'] > 0 else 'AT_OR_BELOW_NULL'))
    v['Q4_label_j'] = ('NA' if c.get('margin_j') is None else
                       ('ABOVE_NULL' if c['margin_j'] > 0 else 'AT_OR_BELOW_NULL'))
    if e6 is not None:
        v['Q1_label'] = ('NF4_FAITHFUL' if (e6.get('pass_tol') and e6.get('argmax_same'))
                         else 'NF4_DEVIANT')
        v['Q1_max_abs_dxh'] = e6.get('max_abs_dxh')
        v['Q1_argmax_same'] = e6.get('argmax_same')
    else:
        v['Q1_label'] = 'NA'
    return v


def main():
    w('Phase 15 (N2h1-alpha-8) 跨模型复算「统一剖面」')
    w('clock %s ; SMOKE=%s ; ARMS=%r' % (time.strftime('%Y-%m-%d %H:%M:%S'), SMOKE, ARMS_SEL))
    w('execution sha256 = %s (seal %s)' % (sha(EXECP), EX['seal_sha8']))
    w('grid: profile_sites=%d alphas=%d W=%d BP=%d XHF=%.2f' % (len(PROFILE), len(ALPHAS), W, BP, XHF))
    w('quant: %s ; max_memory=%s' % (QUANT['scheme'], QUANT['max_memory']))
    w('in 继承参照: d_argmax_4B=%d ; null95 由本 Phase 现场重算' %
      abs(int(INH['MODE_X_13']) - int(INH['MODE_J_13'])))

    order = EX['arm_order']
    if ARMS_SEL:
        sel = set(x.strip() for x in ARMS_SEL.split(','))
        order = [a for a in order if a in sel]
    if SMOKE and not ARMS_SEL:
        order = [a for a in order if a.startswith('A0')]

    results = {}
    if MERGE:
        # 进程隔离模式：从各臂 partial 载入（每臂一个独立进程写出的 _armrec_<arm>.json）
        for arm_id in order:
            pp = os.path.join(P15T, '_armrec_%s.json' % arm_id)
            assert os.path.exists(pp), 'MERGE 缺少 partial: %s' % pp
            results[arm_id] = json.load(io.open(pp, encoding='utf-8'))
            w('  MERGE load %s <- %s (%.1f s)' %
              (arm_id, os.path.basename(pp), results[arm_id].get('elapsed_s', 0)))
        flush_log('_run_all.log')
    else:
        for arm_id in order:
            acfg = EX['arms'][arm_id]
            try:
                results[arm_id] = run_arm(arm_id, acfg)
            except Exception:
                w('!! ARM %s FAILED' % arm_id)
                w(traceback.format_exc())
                results[arm_id] = dict(arm=arm_id, error=traceback.format_exc()[:4000],
                                       model=acfg['model'], role=acfg['role'])
                flush_log('_arm_%s.log' % arm_id)
        flush_log('_run_all.log')

    if SPLIT_PARTIAL and not MERGE:
        w('SPLIT_PARTIAL: 本进程只跑 %s；partial 已写出，跳过 RESULT 组装（用 MERGE=1 合并）' % order)
        return

    # ---- 判决 + result
    verdict = {}
    for arm_id, rec in results.items():
        if 'error' in rec:
            verdict[arm_id] = dict(Q0_device='FAIL', error=True)
        else:
            verdict[arm_id] = per_arm_verdict(rec)
    rep = [a for a in ['A1_glm4-9b-nf4', 'A2_qwen3-14b-nf4'] if a in verdict]
    labels_ok = [a for a in rep if verdict[a].get('Q0_device') == 'PASS']
    q2 = [verdict[a].get('Q2_label') for a in labels_ok]
    q3 = [verdict[a].get('Q3_label') for a in labels_ok]
    joint = dict(
        arms_present=list(results.keys()), arms_used_for_cross_model=labels_ok,
        Q2_joint=('ARGS_GAP_LAYERSTACK' if (q2 and all(x == 'ARGS_GAP_GE3' for x in q2)) else
                  'ARGS_GAP_4B_SPECIFIC' if (q2 and all(x == 'ARGS_GAP_LT3' for x in q2)) else
                  'ARGS_GAP_MIXED' if q2 else 'NA'),
        Q3_joint=('CONC_JUDGE_INVALID_X_ALL' if (q3 and all(x == 'NULL_X_HIGH' for x in q3)) else
                  'CONC_JUDGE_ALIVE_X' if q3 else 'NA'),
    )
    recs = {}
    for arm_id, rec in results.items():
        if 'error' in rec:
            recs[arm_id] = rec
            continue
        recs[arm_id] = {k: v for k, v in rec.items() if k not in ('E4_profile',)}
        recs[arm_id]['E4_per_pair'] = rec['E4_profile']
    RESULT = dict(phase=15, smoke=SMOKE, execution_sha256=sha(EXECP), seal_sha256=EX['seal_sha256'],
                  grid=dict(profile_sites=PROFILE, alphas=ALPHAS, W=W, BP=BP, xh_frac=XHF,
                            cands=list(EX['localize']['cands'])),
                  arms_meta={k: dict(model=v['model'], role=v['role'], config_sha8=v['config_sha8'])
                             for k, v in EX['arms'].items()},
                  E0_selfcheck={k: v.get('E0_selfcheck') for k, v in results.items() if 'error' not in v},
                  E1_capture={k: v.get('E1_capture') for k, v in results.items() if 'error' not in v},
                  E2_full_swap={k: v.get('E2_full_swap') for k, v in results.items() if 'error' not in v},
                  E3_localize={k: v.get('E3_localize') for k, v in results.items() if 'error' not in v},
                  E4_summary={k: v.get('E4_summary') for k, v in results.items() if 'error' not in v},
                  E5_concentration={k: v.get('E5_concentration') for k, v in results.items() if 'error' not in v},
                  E6_calibration={k: v.get('E6_calibration') for k, v in results.items() if 'error' not in v},
                  arms=recs,
                  predictions_check={},
                  verdict=verdict, joint_verdict=joint,
                  floors=FL, inheritance_used=INH,
                  sup_id_per_arm={k: v.get('sup_id_arm') for k, v in results.items() if 'error' not in v},
                  sup_id_ref=SUP_ID_REF,
                  amend1_sha256=sha(AM1P), amend1_sha8=sha(AM1P)[:8],
                  amend1_kind=AM1['kind'],
                  elapsed_total_s=sum(v.get('elapsed_s', 0) for v in results.values()),
                  extra=dict(quant=QUANT))

    # ---- 预注册预测核对
    pc = {}
    ok_arms = [k for k, v in results.items() if 'error' not in v]
    err_arms = [k for k, v in results.items() if 'error' in v]
    dev_ok = bool(ok_arms) and all(v.get('F1b_ok') and v.get('T2_only') and v.get('F4_dims_ok') and
                                   v['E0_selfcheck']['determinism_maxdiff'] == 0 and
                                   v['E0_selfcheck']['hook_effect_maxdiff'] > 0
                                   for k, v in results.items() if 'error' not in v)
    pc['P1'] = dict(pass_=bool(dev_ok), detail='装置自检 ok_arms=%s err_arms=%s' % (ok_arms, err_arms))
    if 'A0_calib_qwen3-4b-nf4' in results and 'error' not in results['A0_calib_qwen3-4b-nf4']:
        e6 = results['A0_calib_qwen3-4b-nf4'].get('E6_calibration') or {}
        pc['P2'] = dict(pass_=bool(e6.get('pass_tol') and e6.get('argmax_same')),
                        detail='A0 max|dxhalf|=%s ; argmax same=%s' %
                               (e6.get('max_abs_dxh'), e6.get('argmax_same')),
                        smoke=SMOKE)
    n95 = {k: (v['E5_concentration'].get('null_x') or {}).get('null95')
           for k, v in results.items() if 'error' not in v}
    pc['P3'] = dict(pass_=all((x is not None and x >= 0.60) for k, x in n95.items() if k in rep) if rep else False,
                    detail={k: x for k, x in n95.items()})
    dag = {k: v['E5_concentration'].get('d_argmax_window')
           for k, v in results.items() if 'error' not in v}
    pc['P4'] = dict(pass_=all((x is not None and x >= FL['ARGS_GAP_MIN']) for k, x in dag.items() if k in rep) if rep else False,
                    detail={k: x for k, x in dag.items()})
    lstar = {k: (v.get('E3_localize') or {}).get('L_star_own')
             for k, v in results.items() if 'error' not in v}
    pc['P5'] = dict(pass_=all((x is not None and x != 6) for k, x in lstar.items() if k in rep) if rep else False,
                    detail={k: x for k, x in lstar.items()})
    rho = {k: v['E5_concentration'].get('spearman_xh_depth')
           for k, v in results.items() if 'error' not in v}
    pc['P6'] = dict(pass_=all((x is not None and x <= FL['RHO_X_MAX']) for k, x in rho.items() if k in rep) if rep else False,
                    detail={k: x for k, x in rho.items()})
    xr = {k: (v.get('E4_summary') or {}).get('XH_RANGE')
          for k, v in results.items() if 'error' not in v}
    band = FL['XH_RANGE_BAND']
    pc['P7'] = dict(pass_=all((x is not None and band[0] <= x <= band[1]) for k, x in xr.items() if k in rep) if rep else False,
                    detail={k: x for k, x in xr.items()})
    RESULT['predictions_check'] = pc

    OUT = os.path.join(P15T, 'result_phase15.json' if not SMOKE else 'result_phase15_smoke.json')
    io.open(OUT, 'w', encoding='utf-8', newline='\n').write(json.dumps(RESULT, ensure_ascii=False, indent=1))
    b = open(OUT, 'rb').read()
    w('')
    w('RESULT -> %s (%d bytes, sha8 %s)' % (OUT, len(b), hashlib.sha256(b).hexdigest()[:8]))
    w('joint_verdict: Q2=%s ; Q3=%s' % (joint['Q2_joint'], joint['Q3_joint']))
    for k, v in pc.items():
        w('  %s : %s  %s' % (k, 'PASS' if v['pass_'] else 'FAIL', str(v.get('detail'))[:150]))
    flush_log('_run_all.log')

    # 每臂报告
    for arm_id, rec in results.items():
        if 'error' in rec:
            continue
        rep_lines = ['# Phase 15 report %s (%s)' % (arm_id, rec['model']),
                     'role: %s' % rec['role'],
                     'cfg: %s' % rec['cfg'],
                     'E0: %s' % rec['E0_selfcheck'],
                     'E1: %s' % rec['E1_capture'],
                     'E2: %s' % rec['E2_full_swap'],
                     'E3: %s' % {k: v for k, v in rec['E3_localize'].items() if k != 'U_meta'},
                     'E4_summary: %s' % rec['E4_summary'],
                     'E5: %s' % rec['E5_concentration'],
                     'E6: %s' % (rec.get('E6_calibration') or 'NA'),
                     'verdict: %s' % verdict.get(arm_id)]
        io.open(os.path.join(P15T, 'n2h1a8_report_%s.txt' % arm_id), 'w',
                encoding='utf-8', newline='\n').write('\n'.join(rep_lines) + '\n')


if __name__ == '__main__':
    main()
