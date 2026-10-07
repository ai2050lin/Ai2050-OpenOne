# -*- coding: utf-8 -*-
"""
Phase 16 (N2h1-alpha-9) 主脚本：写入窗原点化剖面 + 集中度统计量重设计。
三臂（同一 nf4 口径，串行）：
  A0_calib_qwen3-4b-nf4 : 量化保真校准（与 Phase 12 bf16 已发表量逐位点比对）
  A1_glm4-9b-nf4        : 跨家族 untied 复算
  A2_qwen3-14b-nf4      : 同家族 untied 规模放大复算
每臂流程：E0 装置自检 -> E1 capture(41) -> E2 FULL_SWAP -> E3 独立写入窗定位 -> E4 单点替换族双坐标剖面
          -> E5 集中度（legacy 域 / 主域旧量 / 主域新量）+ E7 可达性剖面 -> (A0) E6 量化保真校准

本 Phase 相对 Phase 15 的**唯一**改动：
  1) profile_sites 下探到 1..5（使 A1 的 L*=3 / A2 的 L*=4 落入剖面域）；
  2) 以可达性掩膜 REACH = {ell : rho(ell) >= UNREACH_y} 定义主域，使写入窗成为主域左端点；
  3) 集中度重设计：以顺序敏感的 (com_layer, span_k) 替换极值型 top3_share，并做**双边**置换检验；
     同时保留 legacy 6..34 域上的旧量用于 P2 冻结锚逐位复现。
装置（hooks / BASE / FULL_SWAP / E3 localize / B_cat）与 Phase 15 逐字节相同。
判据 Q0-Q5 全部预先冻结在 seal 中；本脚本只执行不做判断。
用法：SMOKE=1 python n2h1a9_writewin_origin_profile.py   (仅 A0，浅+深混合小网格)
      python n2h1a9_writewin_origin_profile.py           (三臂正式；建议 SPLIT_PARTIAL=1 逐臂进程隔离)
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
P16 = os.path.join(ROOT, 'tests', 'deepseek', 'Phase16')
P16T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase16')
P15T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase15')
EXECP = os.path.join(P16T, 'execution_phase16.json')
SEALP = os.path.join(P16T, 'N2h1a9_design_seal.json')
ANCHP = os.path.join(P15T, 'result_phase15.json')
os.makedirs(P16T, exist_ok=True)
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

# Phase 15 result 作为**冻结锚**（P2 逐位复现对象）；其 sha 冻结在 seal/exec 中
assert os.path.exists(ANCHP), 'ANCHOR_MISSING: %s' % ANCHP
_ab = open(ANCHP, 'rb').read()
assert hashlib.sha256(_ab).hexdigest() == EX['anchor_result_sha256'], \
    'DRIFT: Phase 15 result 锚漂移'
ANCH_ALL = json.loads(_ab.decode('utf-8'))
ANCH = EX['anchor_values']

# amend1（锚判据分层，见 tests/deepseek_temp/Phase16/N2h1a9_design_seal_amend1.json）
AM1P = os.path.join(P16T, 'N2h1a9_design_seal_amend1.json')
assert os.path.exists(AM1P), 'AMEND1_MISSING: %s' % AM1P
AM1 = json.load(io.open(AM1P, encoding='utf-8'))
assert AM1['amend_of_seal_sha256'] == EX['seal_sha256'], 'DRIFT: amend1 指向的 seal 与本 exec 不一致'

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
    # SMOKE 必须覆盖三类位点：浅端 (1,2,3，检验写入窗下探)、写入窗本身 (6)、深端 (30,34)
    ALPHAS = [0.0, 0.5, 1.0]
    PROFILE = [1, 2, 3, 6, 30, 34]
    CANDS = [1, 4, 6, 20]
    BP = 200

_log = []


def w(s=''):
    _log.append(str(s))
    print(s)
    sys.stdout.flush()


def flush_log(name):
    io.open(os.path.join(P16T, name), 'w', encoding='utf-8', newline='\n').write('\n'.join(_log) + '\n')


def F3(v, nd=3):
    """None-安全的数值格式化（叙述行一律走它，避免降级分支 KeyError/TypeError）。"""
    try:
        return ('%.*f' % (nd, float(v))) if (v is not None and np.isfinite(float(v))) else 'None'
    except Exception:
        return 'None'


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


# ---------------------------------------------------------------- 集中度重设计（Phase 16）
# 结构性说明：置换零假设**保留 jump 的多重集**，故任何只依赖多重集的量（谱熵、max|d|/mean|d|、
# 参与比 sum|d|^2/(sum|d|)^2）在零假设下**恒等于观测值** -> 双边检验必得 p=1，属于**结构性退化族**。
# 可检验的量必须**顺序敏感**。本 Phase 用两个：
#   com_layer : |delta| 在**物理层号**轴上的质心（单位：层）。尺度无关；定义在物理轴上 => 对网格
#               加密/下探不变（这是 Phase 15 用 jump 序号归一化的版本所缺的性质）。
#   span_k    : 最大的 k 个 |delta| 所在步的跨度 / (n-1)。尺度无关；小 = 主变挤在少数相邻步。

def stat_com_layer(jumps, sites):
    j = np.asarray(jumps, float)
    if len(j) == 0 or len(sites) != len(j) + 1:
        return None, None
    mid = (np.asarray(sites, float)[:-1] + np.asarray(sites, float)[1:]) / 2.0
    a = np.abs(j)
    den = float(a.sum())
    if not np.isfinite(den) or den <= 1e-12:
        return None, None
    com_abs = float((a * mid).sum() / den)
    ds = float(j.sum())
    com_sgn = float((j * mid).sum() / ds) if abs(ds) > 1e-12 else None
    return com_abs, com_sgn


def stat_span_k(jumps, k=3):
    j = np.asarray(jumps, float)
    n = len(j)
    if n < k or not np.isfinite(j).all():
        return None
    idx = np.argsort(-np.abs(j))[:k]
    return float(idx.max() - idx.min()) / max(n - 1, 1)


def _ns_empty(n_bp, reason):
    """降级分支也必须返回**同一个 schema**（SMOKE 实测教训：缺键会在叙述行 KeyError）。"""
    return dict(BP=n_bp, n_ok=0, reason=reason,
                obs_com=None, com_p5=None, com_p95=None, com_tail=None,
                obs_com_signed=None, obs_span=None, span_p5=None, span_p95=None,
                span_tail=None, span_degenerate=None, com_ci=None, span_ci=None)


def perm_null_new(jumps, sites, rng_obj, n_bp, k=3):
    """同一置换协议（多重集随机排列）下 com_layer 与 span_k 的**双边**分位。"""
    j = np.asarray(jumps, float)
    n = len(j)
    if n < k + 1 or not np.isfinite(j).all():
        return _ns_empty(n_bp, 'bad_input(n=%d)' % n)
    obs_com, obs_com_sgn = stat_com_layer(j, sites)
    obs_span = stat_span_k(j, k)
    cs = np.full(n_bp, np.nan)
    ss = np.full(n_bp, np.nan)
    for b in range(n_bp):
        p = j[rng_obj.permutation(n)]
        c, _ = stat_com_layer(p, sites)
        cs[b] = c if c is not None else np.nan
        vv = stat_span_k(p, k)
        ss[b] = vv if vv is not None else np.nan
    fc = cs[np.isfinite(cs)]
    fs = ss[np.isfinite(ss)]
    if len(fc) == 0 or len(fs) == 0:
        return _ns_empty(n_bp, 'all_nan')
    p5c, p95c = float(np.percentile(fc, 5)), float(np.percentile(fc, 95))
    p5s, p95s = float(np.percentile(fs, 5)), float(np.percentile(fs, 95))
    tc = ('low' if (obs_com is not None and obs_com <= p5c) else
          'high' if (obs_com is not None and obs_com >= p95c) else 'none')
    ts = ('low' if (obs_span is not None and obs_span <= p5s) else
          'high' if (obs_span is not None and obs_span >= p95s) else 'none')
    return dict(BP=n_bp, n_ok=int(len(fc)),
                obs_com=obs_com, com_p5=p5c, com_p95=p95c, com_tail=tc,
                obs_com_signed=obs_com_sgn,
                obs_span=obs_span, span_p5=p5s, span_p95=p95s, span_tail=ts,
                span_degenerate=bool(p95s - p5s < 1e-12),
                com_ci=dict(lo=p5c, hi=p95c, med=float(np.median(fc))),
                span_ci=dict(lo=p5s, hi=p95s, med=float(np.median(fs))))


def slice_sites(F, sites, keep):
    """按位点集合取子向量（保持原顺序）。"""
    idx = [i for i, sv in enumerate(sites) if int(sv) in keep]
    return np.array([F[i] for i in idx], float), [int(sites[i]) for i in idx]


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
    LEGACY_SET = set(int(x) for x in EX['profile_sites_legacy'])
    _li = [i for i, sv in enumerate(PROFILE) if int(sv) in LEGACY_SET]
    xh_leg = np.array([xh[i] for i in _li], float)
    Jv_leg = np.array([Jv[i] for i in _li], float)
    sites_leg = [int(PROFILE[i]) for i in _li]
    w('  E4b legacy 子域 n=%d (sites %s..%s)' % (len(sites_leg), sites_leg[0], sites_leg[-1]))
    rec['E4_summary'] = dict(sites=PROFILE, alphas=ALPHAS,
                             xhalf=[float(v) for v in xh], J=[float(v) for v in Jv],
                             n_finite_x=int(np.isfinite(xh).sum()),
                             n_finite_j=int(np.isfinite(Jv).sum()),
                             XH_RANGE=(float(np.nanmax(xh) - np.nanmin(xh)) if np.isfinite(xh).any() else None),
                             XH_RANGE_legacy=(float(np.nanmax(xh_leg) - np.nanmin(xh_leg)) if np.isfinite(xh_leg).any() else None),
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

    # --- E7 可达性剖面（rho(ell) = Y(ell, alpha=1)）与可达域 REACH
    a1i = list(ALPHAS).index(1.0) if 1.0 in ALPHAS else None
    rho = (Y[:, a1i] if a1i is not None else np.full(len(PROFILE), np.nan))
    UNR = float(FL['UNREACH_y'])
    REACH = [int(sv) for i, sv in enumerate(PROFILE)
             if np.isfinite(rho[i]) and float(rho[i]) >= UNR]
    EXCL = [int(sv) for sv in PROFILE if int(sv) not in REACH]
    _cross = [int(sv) for i, sv in enumerate(PROFILE)
              if np.isfinite(rho[i]) and float(rho[i]) >= 0.5]
    ell_reach = (min(_cross) if _cross else None)
    rec['E7_reach'] = dict(sites=[int(sv) for sv in PROFILE], rho=[float(v) for v in rho],
                           unreach_y=UNR, reach=REACH, excluded=EXCL,
                           ell_reach=ell_reach, full_swap=FULL_SWAP)
    w('  E7 可达性: REACH n=%d ; EXCL(%d)=%s ; ell_reach(rho>=0.5)=%s' %
      (len(REACH), len(EXCL), EXCL, ell_reach))
    w('     rho: %s' % ' '.join('L%d:%.4f' % (PROFILE[i], rho[i]) for i in range(len(PROFILE))))

    # --- E5 集中度（三口径）
    #   legacy_domain : 与 Phase 15 完全相同的 6..34 子域 + 旧量 top3_share + 冻结种子 -> P2 锚复现
    #   main_domain   : 可达域 REACH 上的旧量（对照）
    #   new_stat      : 可达域 REACH 上的新量 (com_layer, span_k)，双边分位
    RSET = set(REACH)
    xh_m, sites_m = slice_sites(xh, PROFILE, RSET)
    Jv_m, _ = slice_sites(Jv, PROFILE, RSET)

    def _conc(F, sites_v, rng_obj):
        share, axw, jm = conc_hat(F, W)
        rng_v = (float(np.nanmax(F) - np.nanmin(F)) if (len(F) and np.isfinite(F).any())
                 else float('nan'))
        nn = (perm_null(jm, rng_obj, W, BP, rng_v)
              if (np.isfinite(jm).all() and np.isfinite(rng_v)) else dict(BP=BP, null95=None))
        mg = ((share - nn['null95']) if (share is not None and nn.get('null95') is not None)
              else None)
        return dict(sites=[int(v) for v in sites_v], F=[float(v) for v in F],
                    jumps=[float(v) for v in jm], top3=share, argmax_w=axw, range=rng_v,
                    null=nn, margin=mg,
                    win_sem=(None if axw is None else
                             dict(w=axw, a=int(sites_v[axw]),
                                  b=int(sites_v[min(axw + W, len(sites_v) - 1)]))),
                    spearman_depth=spearman(list(F), [int(v) for v in sites_v]))

    rngA = np.random.default_rng(SEED + 13)   # 与 Phase 15 A0 旧量完全同一协议
    rngB = np.random.default_rng(SEED + 29)
    rngC = np.random.default_rng(SEED + 41)
    rngD = np.random.default_rng(SEED + 53)
    leg_x = _conc(xh_leg, sites_leg, rngA)
    leg_j = _conc(Jv_leg, sites_leg, rngB)
    main_x = _conc(xh_m, sites_m, rngC)
    main_j = _conc(Jv_m, sites_m, rngD)
    new_x = perm_null_new(np.diff(xh_m), sites_m, np.random.default_rng(SEED + 61), BP, W)
    new_j = perm_null_new(np.diff(Jv_m), sites_m, np.random.default_rng(SEED + 67), BP, W)
    rec['E5_concentration'] = dict(
        W=W, alpha_grid=ALPHAS,
        legacy_domain=dict(
            x=leg_x, j=leg_j,
            d_argmax_window=(abs(int(leg_x['argmax_w']) - int(leg_j['argmax_w']))
                             if (leg_x['argmax_w'] is not None and leg_j['argmax_w'] is not None)
                             else None)),
        main_domain=dict(
            x=main_x, j=main_j,
            d_argmax_window=(abs(int(main_x['argmax_w']) - int(main_j['argmax_w']))
                             if (main_x['argmax_w'] is not None and main_j['argmax_w'] is not None)
                             else None)),
        new_stat=dict(x=new_x, j=new_j, k=W))
    w('  E5 legacy 域(%d 步) top3_x=%s (w=%s) top3_j=%s (w=%s)' %
      (len(leg_x['jumps']), leg_x['top3'], leg_x['argmax_w'], leg_j['top3'], leg_j['argmax_w']))
    w('     legacy null95_x=%s margin_x=%s ; null95_j=%s margin_j=%s' %
      (leg_x['null'].get('null95'), leg_x['margin'], leg_j['null'].get('null95'), leg_j['margin']))
    w('  E5 主域(%d 步) 旧量 top3_x=%s(w=%s) top3_j=%s(w=%s) -> sig=%s' %
      (len(main_x['jumps']), main_x['top3'], main_x['argmax_w'], main_j['top3'], main_j['argmax_w'],
       [k for k in ('x', 'j') if (dict(x=main_x, j=main_j)[k]['margin'] or -1) > 0]))
    w('  E5 新量 com_x=%s [%s, %s] tail=%s ; com_j=%s [%s, %s] tail=%s' %
      (F3(new_x.get('obs_com')), F3(new_x.get('com_p5')), F3(new_x.get('com_p95')),
       new_x.get('com_tail'), F3(new_j.get('obs_com')), F3(new_j.get('com_p5')),
       F3(new_j.get('com_p95')), new_j.get('com_tail')))
    w('     span_x=%s tail=%s ; span_j=%s tail=%s ; com_sep=%s 层' %
      (F3(new_x.get('obs_span'), 4), new_x.get('span_tail'),
       F3(new_j.get('obs_span'), 4), new_j.get('span_tail'),
       F3((None if (new_x.get('obs_com') is None or new_j.get('obs_com') is None)
           else new_x['obs_com'] - new_j['obs_com']))))
    w('     新量降级原因: x=%s / j=%s' % (new_x.get('reason'), new_j.get('reason')))

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
        share_x = leg_x['top3']; axw_x = leg_x['argmax_w']
        _L6 = [int(s) for s in PROFILE if int(s) in XH12]
        share_x_12, axw_x_12v, _ = conc_hat(np.array([XH12[s] for s in _L6], float), W)
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
            XH_RANGE_nf4=rec['E4_summary']['XH_RANGE_legacy'], XH_RANGE_bf16=float(INH['XH_RANGE_12']))
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
           rec['E4_summary']['XH_RANGE_legacy'] or float('nan'), rec['E6_calibration']['XH_RANGE_bf16']))
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
        pp = os.path.join(P16T, '_armrec16_%s.json' % arm_id)
        io.open(pp, 'w', encoding='utf-8', newline='\n').write(json.dumps(rec, ensure_ascii=False))
        w('  SPLIT_PARTIAL -> %s (%d B)' % (os.path.basename(pp), os.path.getsize(pp)))
    return rec


# ---------------------------------------------------------------- 判决
def per_arm_verdict(rec, anch):
    c = rec.get('E5_concentration') or {}
    e6 = rec.get('E6_calibration')
    e3 = rec.get('E3_localize') or {}
    e7 = rec.get('E7_reach') or {}
    ld = c.get('legacy_domain') or {}
    md = c.get('main_domain') or {}
    ns = c.get('new_stat') or {}
    v = {}
    v['Q0_device'] = ('PASS' if (rec.get('F1b_ok') and rec.get('T2_only') and rec.get('F4_dims_ok') and
                                 rec.get('F2_base_ok') and
                                 rec['E0_selfcheck']['determinism_maxdiff'] == 0 and
                                 rec['E0_selfcheck']['hook_effect_maxdiff'] > 0) else 'FAIL')
    # ---- Q1 冻结锚复现（legacy 6..34 域）
    lx = ((ld.get('x') or {}).get('F') or [])
    lj = ((ld.get('j') or {}).get('F') or [])
    ls = ((ld.get('x') or {}).get('sites') or [])
    axh = {str(k): float(val) for k, val in (anch.get('xhalf_by_site') or {}).items()}
    aJ = {str(k): float(val) for k, val in (anch.get('J_by_site') or {}).items()}
    dx = [abs(lx[i] - axh[str(ls[i])]) for i in range(len(ls))
          if i < len(lx) and np.isfinite(lx[i]) and str(ls[i]) in axh]
    dJ = [abs(lj[i] - aJ[str(ls[i])]) / max(abs(aJ[str(ls[i])]), 1e-9) for i in range(len(ls))
          if i < len(lj) and np.isfinite(lj[i]) and str(ls[i]) in aJ]
    v['Q1_n_sites'] = len(ls)
    v['Q1_max_abs_dxh_leg'] = (float(max(dx)) if dx else None)
    v['Q1_max_rel_dJ_leg'] = (float(max(dJ)) if dJ else None)
    v['Q1_argmax_same_leg'] = bool(ls and (ld.get('x') or {}).get('argmax_w') is not None and
                                   (ld.get('x') or {}).get('argmax_w') == anch.get('legacy_argmax_w_x') and
                                   (ld.get('j') or {}).get('argmax_w') is not None and
                                   (ld.get('j') or {}).get('argmax_w') == anch.get('legacy_argmax_w_j'))
    v['Q1_label'] = 'RECON_DRIFT'
    _t3x = (ld.get('x') or {}).get('top3')
    _t3j = (ld.get('j') or {}).get('top3')
    _dt3x = (abs(_t3x - float(anch['legacy_top3_x']))
             if (_t3x is not None and anch.get('legacy_top3_x') is not None) else None)
    _dt3j = (abs(_t3j - float(anch['legacy_top3_j']))
             if (_t3j is not None and anch.get('legacy_top3_j') is not None) else None)
    v['Q1_dtop3_x'] = _dt3x
    v['Q1_dtop3_j'] = _dt3j
    _fn = AM1['floors']
    v['Q1_dxh_by_site'] = {str(ls[i]): float(dx[i]) for i in range(min(len(dx), len(ls)))}
    _base_ok = bool(dx and dJ and _dt3x is not None and _dt3j is not None and
                    max(dJ) <= _fn['RECON_TOL_J_REL'] and _dt3x <= _fn['RECON_TOL_TOP3'] and
                    _dt3j <= _fn['RECON_TOL_TOP3'] and v['Q1_argmax_same_leg'])
    if _base_ok and max(dx) <= _fn['RECON_TOL_XH_STRICT']:
        v['Q1_label'] = 'RECON_OK'
    elif _base_ok and max(dx) <= _fn['RECON_TOL_XH_LOOSE']:
        v['Q1_label'] = 'RECON_OK_LOOSE'
    else:
        v['Q1_label'] = 'RECON_DRIFT'
    # ---- Q2 可达域左端点 == 写入窗
    v['Q2_L_star_own'] = e3.get('L_star_own')
    v['Q2_ell_reach'] = e7.get('ell_reach')
    v['Q2_excluded'] = e7.get('excluded')
    v['Q2_label'] = ('NA' if (v['Q2_L_star_own'] is None or v['Q2_ell_reach'] is None) else
                     ('REACH_EQ_WRITEWIN' if int(v['Q2_ell_reach']) == int(v['Q2_L_star_own'])
                      else 'REACH_OFFSET(d=%+d)' % (int(v['Q2_ell_reach']) - int(v['Q2_L_star_own']))))
    # ---- Q3 写入窗入域
    _sites_all = e7.get('sites') or []
    v['Q3_label'] = ('NA' if (v['Q2_L_star_own'] is None or not _sites_all) else
                     ('WIN_IN_DOMAIN' if min(_sites_all) <= int(v['Q2_L_star_own']) <= max(_sites_all)
                      else 'WIN_OUT_OF_DOMAIN'))
    # ---- Q4 物理深度质心分离
    cx = (ns.get('x') or {}).get('obs_com')
    cj = (ns.get('j') or {}).get('obs_com')
    v['Q4_com_x'] = cx
    v['Q4_com_j'] = cj
    v['Q4_sep'] = (float(cx) - float(cj)) if (cx is not None and cj is not None) else None
    v['Q4_label'] = ('NA' if v['Q4_sep'] is None else
                     ('CENTROID_SEPARATED' if v['Q4_sep'] >= FL['CENTROID_SEP_MIN']
                      else 'CENTROID_OVERLAP'))
    v['Q4_com_after_win_x'] = (float(cx) - int(v['Q2_L_star_own'])
                               if (cx is not None and v['Q2_L_star_own'] is not None) else None)
    # ---- Q5 重设计有效性（主域上：旧量单边 vs 新量双边）
    _md = dict(x=md.get('x') or {}, j=md.get('j') or {})
    _ns = dict(x=ns.get('x') or {}, j=ns.get('j') or {})
    v['Q5_old_sig'] = [k for k in ('x', 'j')
                       if _md[k].get('margin') is not None and _md[k]['margin'] > 0]
    v['Q5_new_sig'] = [k for k in ('x', 'j')
                       if _ns[k].get('com_tail') not in (None, 'none')]
    v['Q5_new_sig_any'] = [k for k in ('x', 'j')
                           if _ns[k].get('com_tail') not in (None, 'none')
                           or _ns[k].get('span_tail') not in (None, 'none')]
    v['Q5_old_margins'] = {k: _md[k].get('margin') for k in ('x', 'j')}
    v['Q5_new_tails'] = {k: _ns[k].get('com_tail') for k in ('x', 'j')}
    v['Q5_new_spans'] = {k: _ns[k].get('obs_span') for k in ('x', 'j')}
    if len(v['Q5_new_sig']) >= len(v['Q5_old_sig']) and len(v['Q5_new_sig']) >= FL['NEW_NONDEG_MIN']:
        v['Q5_label'] = 'STAT_REDESIGN_EFFECTIVE'
    elif len(v['Q5_new_sig']) >= len(v['Q5_old_sig']):
        v['Q5_label'] = 'STAT_REDESIGN_PARTIAL'
    else:
        v['Q5_label'] = 'STAT_REDESIGN_EQUIVALENT'
    if e6 is not None:
        v['Q6_e6_label'] = ('NF4_FAITHFUL' if e6.get('pass_tol') else 'NF4_DEVIANT')
        v['Q6_e6_max_abs_dxh'] = e6.get('max_abs_dxh')
    return v


def main():
    w('Phase 16 (N2h1-alpha-9) 写入窗原点化剖面 + 集中度重设计')
    w('clock %s ; SMOKE=%s ; ARMS=%r' % (time.strftime('%Y-%m-%d %H:%M:%S'), SMOKE, ARMS_SEL))
    w('execution sha256 = %s (seal %s)' % (sha(EXECP), EX['seal_sha8']))
    w('grid: profile_sites=%d alphas=%d W=%d BP=%d XHF=%.2f' % (len(PROFILE), len(ALPHAS), W, BP, XHF))
    w('quant: %s ; max_memory=%s' % (QUANT['scheme'], QUANT['max_memory']))
    w('冻结锚: Phase 15 result sha8 = %s ; 三臂 L*_own=%s' %
      (EX['anchor_result_sha256'][:8], {k: ANCH[k]['L_star_own'] for k in EX['arm_order']}))
    w('新增网格: 浅端 %s ; 主域 = REACH(rho>=%.2f) ; 新量 = com_layer / span_k' %
      ([s for s in PROFILE if s < 6], FL['UNREACH_y']))

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
            pp = os.path.join(P16T, '_armrec16_%s.json' % arm_id)
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
            verdict[arm_id] = per_arm_verdict(rec, ANCH[arm_id])
    rep = [a for a in ['A1_glm4-9b-nf4', 'A2_qwen3-14b-nf4'] if a in verdict]
    labels_ok = [a for a in verdict if verdict[a].get('Q0_device') == 'PASS']
    q1 = [verdict[a].get('Q1_label') for a in labels_ok]
    q2 = [verdict[a].get('Q2_label') for a in labels_ok]
    q3 = [verdict[a].get('Q3_label') for a in labels_ok]
    q4 = [verdict[a].get('Q4_label') for a in labels_ok]
    q5o = [len(verdict[a].get('Q5_old_sig') or []) for a in labels_ok]
    q5n = [len(verdict[a].get('Q5_new_sig') or []) for a in labels_ok]
    joint = dict(
        arms_present=list(results.keys()), arms_used_for_cross_model=labels_ok,
        Q1_joint=('ANCHOR_ROBUST' if (q1 and all(x == 'RECON_OK' for x in q1)) else
                  'ANCHOR_ROBUST_TIERED' if (q1 and all(x in ('RECON_OK', 'RECON_OK_LOOSE') for x in q1)
                                             and sum(1 for x in q1 if x == 'RECON_OK') >= 2) else
                  'ANCHOR_PARTIAL' if (q1 and any(x in ('RECON_OK', 'RECON_OK_LOOSE') for x in q1)) else
                  'ANCHOR_FAIL' if q1 else 'NA'),
        Q2_joint=('REACH_IDENTITY_ROBUST' if (q2 and all(x == 'REACH_EQ_WRITEWIN' for x in q2)) else
                  'REACH_IDENTITY_PARTIAL' if (q2 and any(x == 'REACH_EQ_WRITEWIN' for x in q2)) else
                  'REACH_IDENTITY_FAIL' if q2 else 'NA'),
        Q3_joint=('WIN_IN_DOMAIN_ALL' if (q3 and all(x == 'WIN_IN_DOMAIN' for x in q3)) else
                  'WIN_DOMAIN_MIXED' if q3 else 'NA'),
        Q4_joint=('CENTROID_SEPARATED_ALL' if (q4 and all(x == 'CENTROID_SEPARATED' for x in q4)) else
                  'CENTROID_PARTIAL' if (q4 and any(x == 'CENTROID_SEPARATED' for x in q4)) else
                  'CENTROID_OVERLAP' if q4 else 'NA'),
        Q5_joint=('STAT_REDESIGN_EFFECTIVE' if (q5n and all(n >= o for n, o in zip(q5n, q5o))
                                                and max(q5n) >= FL['NEW_NONDEG_MIN']) else
                  'STAT_REDESIGN_EQUIVALENT' if q5n else 'NA'),
        Q5_counts=dict(old_sig=q5o, new_sig=q5n),
    )
    recs = {}
    for arm_id, rec in results.items():
        if 'error' in rec:
            recs[arm_id] = rec
            continue
        recs[arm_id] = {k: v for k, v in rec.items() if k not in ('E4_profile',)}
        recs[arm_id]['E4_per_pair'] = rec['E4_profile']
    RESULT = dict(phase=16, smoke=SMOKE, execution_sha256=sha(EXECP), seal_sha256=EX['seal_sha256'],
                  grid=dict(profile_sites=PROFILE, profile_sites_legacy=EX['profile_sites_legacy'],
                            alphas=ALPHAS, W=W, BP=BP, xh_frac=XHF,
                            cands=list(EX['localize']['cands'])),
                  arms_meta={k: dict(model=v['model'], role=v['role'], config_sha8=v['config_sha8'])
                             for k, v in EX['arms'].items()},
                  E0_selfcheck={k: v.get('E0_selfcheck') for k, v in results.items() if 'error' not in v},
                  E1_capture={k: v.get('E1_capture') for k, v in results.items() if 'error' not in v},
                  E2_full_swap={k: v.get('E2_full_swap') for k, v in results.items() if 'error' not in v},
                  E3_localize={k: v.get('E3_localize') for k, v in results.items() if 'error' not in v},
                  E4_summary={k: v.get('E4_summary') for k, v in results.items() if 'error' not in v},
                  E5_concentration={k: v.get('E5_concentration') for k, v in results.items() if 'error' not in v},
                  E7_reach={k: v.get('E7_reach') for k, v in results.items() if 'error' not in v},
                  E6_calibration={k: v.get('E6_calibration') for k, v in results.items() if 'error' not in v},
                  arms=recs,
                  predictions_check={},
                  verdict=verdict, joint_verdict=joint,
                  floors=FL, inheritance_used=INH,
                  sup_id_per_arm={k: v.get('sup_id_arm') for k, v in results.items() if 'error' not in v},
                  sup_id_ref=SUP_ID_REF,
                  anchor_result_sha256=EX['anchor_result_sha256'],
                  anchor_result_sha8=EX['anchor_result_sha256'][:8],
                  anchor_phase=15,
                  amend1_sha256=hashlib.sha256(open(AM1P, 'rb').read()).hexdigest(),
                  amend1_sha8=hashlib.sha256(open(AM1P, 'rb').read()).hexdigest()[:8],
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
    # P2 冻结锚逐位复现（三臂 legacy 6..34；amend1 分层判据）
    _r1 = {k: verdict[k].get('Q1_label') for k in ok_arms}
    _n_ok = sum(1 for k in ok_arms if verdict[k].get('Q1_label') == 'RECON_OK')
    _n_loose = sum(1 for k in ok_arms if verdict[k].get('Q1_label') == 'RECON_OK_LOOSE')
    pc['P2'] = dict(pass_=bool(len(ok_arms) == 3 and
                               all(v in ('RECON_OK', 'RECON_OK_LOOSE') for v in _r1.values()) and
                               _n_ok >= 2),
                    detail=dict(labels=_r1, n_strict_ok=_n_ok, n_loose=_n_loose,
                                max_abs_dxh={k: verdict[k].get('Q1_max_abs_dxh_leg') for k in ok_arms},
                                max_rel_dJ={k: verdict[k].get('Q1_max_rel_dJ_leg') for k in ok_arms},
                                dtop3={k: [verdict[k].get('Q1_dtop3_x'), verdict[k].get('Q1_dtop3_j')] for k in ok_arms},
                                argmax_same={k: verdict[k].get('Q1_argmax_same_leg') for k in ok_arms},
                                dxh_by_site={k: verdict[k].get('Q1_dxh_by_site') for k in ok_arms},
                                tiers=AM1['floors'], p2_criterion=AM1['p2_criterion']))
    # P3 写入窗入域且重算与 Phase 15 一致
    _ls_now = {k: verdict[k].get('Q2_L_star_own') for k in ok_arms}
    _ls_15 = {k: int(ANCH[k]['L_star_own']) for k in ok_arms}
    _dom = {k: verdict[k].get('Q3_label') for k in ok_arms}
    pc['P3'] = dict(pass_=bool(ok_arms and all(_ls_now[k] == _ls_15[k] for k in ok_arms)
                               and all(_dom[k] == 'WIN_IN_DOMAIN' for k in ok_arms)),
                    detail=dict(L_star_now=_ls_now, L_star_15=_ls_15, domain=_dom,
                                profile_lo=int(min(PROFILE)), profile_hi=int(max(PROFILE))))
    # P4 可达域左端点 == 写入窗（严格相等）
    _er = {k: verdict[k].get('Q2_ell_reach') for k in ok_arms}
    pc['P4'] = dict(pass_=bool(ok_arms and all(verdict[k].get('Q2_label') == 'REACH_EQ_WRITEWIN'
                                               for k in ok_arms)),
                    detail=dict(ell_reach=_er, L_star=_ls_now, excluded={k: verdict[k].get('Q2_excluded') for k in ok_arms},
                                labels={k: verdict[k].get('Q2_label') for k in ok_arms}))
    # P5 新量不劣于旧量
    _n_old = sum(len(verdict[k].get('Q5_old_sig') or []) for k in ok_arms)
    _n_new = sum(len(verdict[k].get('Q5_new_sig') or []) for k in ok_arms)
    pc['P5'] = dict(pass_=bool(_n_new >= _n_old and _n_new >= FL['NEW_NONDEG_MIN']),
                    detail=dict(n_old_sig=_n_old, n_new_sig=_n_new, min_required=FL['NEW_NONDEG_MIN'],
                                old_margins={k: verdict[k].get('Q5_old_margins') for k in ok_arms},
                                new_tails={k: verdict[k].get('Q5_new_tails') for k in ok_arms}))
    # P6 物理深度质心分离 >= 4.0 层（3/3）
    _sep = {k: verdict[k].get('Q4_sep') for k in ok_arms}
    pc['P6'] = dict(pass_=bool(ok_arms and all((v is not None and v >= FL['CENTROID_SEP_MIN'])
                                               for v in _sep.values())),
                    detail=dict(sep_layers=_sep, com_x={k: verdict[k].get('Q4_com_x') for k in ok_arms},
                                com_j={k: verdict[k].get('Q4_com_j') for k in ok_arms},
                                threshold=FL['CENTROID_SEP_MIN']))
    # P7 xhalf 质心位于写入窗之后 >= 5 层（>= 2/3）
    _aft = {k: verdict[k].get('Q4_com_after_win_x') for k in ok_arms}
    _naft = sum(1 for v in _aft.values() if v is not None and v >= FL['CENTROID_AFTER_WIN_MIN'])
    pc['P7'] = dict(pass_=bool(_naft >= 2 and len(_aft) == 3),
                    detail=dict(after_win_layers=_aft, n_pass=_naft, n_arms=len(_aft),
                                threshold=FL['CENTROID_AFTER_WIN_MIN']))
    RESULT['predictions_check'] = pc

    OUT = os.path.join(P16T, 'result_phase16.json' if not SMOKE else 'result_phase16_smoke.json')
    io.open(OUT, 'w', encoding='utf-8', newline='\n').write(json.dumps(RESULT, ensure_ascii=False, indent=1))
    b = open(OUT, 'rb').read()
    w('')
    w('RESULT -> %s (%d bytes, sha8 %s)' % (OUT, len(b), hashlib.sha256(b).hexdigest()[:8]))
    w('joint_verdict: Q1=%s ; Q2=%s ; Q3=%s ; Q4=%s ; Q5=%s' %
      (joint['Q1_joint'], joint['Q2_joint'], joint['Q3_joint'], joint['Q4_joint'], joint['Q5_joint']))
    for k, v in pc.items():
        w('  %s : %s  %s' % (k, 'PASS' if v['pass_'] else 'FAIL', str(v.get('detail'))[:150]))
    flush_log('_run_all.log')

    # 每臂报告
    for arm_id, rec in results.items():
        if 'error' in rec:
            continue
        rep_lines = ['# Phase 16 report %s (%s)' % (arm_id, rec['model']),
                     'role: %s' % rec['role'],
                     'cfg: %s' % rec['cfg'],
                     'E0: %s' % rec['E0_selfcheck'],
                     'E1: %s' % rec['E1_capture'],
                     'E2: %s' % rec['E2_full_swap'],
                     'E3: %s' % {k: v for k, v in rec['E3_localize'].items() if k != 'U_meta'},
                     'E4_summary: %s' % rec['E4_summary'],
                     'E5: %s' % rec['E5_concentration'],
                     'E7: %s' % rec.get('E7_reach'),
                     'E6: %s' % (rec.get('E6_calibration') or 'NA'),
                     'verdict: %s' % verdict.get(arm_id)]
        io.open(os.path.join(P16T, 'n2h1a9_report_%s.txt' % arm_id), 'w',
                encoding='utf-8', newline='\n').write('\n'.join(rep_lines) + '\n')


if __name__ == '__main__':
    main()
