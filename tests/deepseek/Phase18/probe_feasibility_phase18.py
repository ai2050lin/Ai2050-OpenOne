# -*- coding: utf-8 -*-
"""
Phase 18 (N2h1-alpha-11) 可行性探针 —— 逐层组件「行为」预算。

背景（为什么需要本 Phase）：
  Phase 17 把 Phase 8 的向量预算 share_v 从单层 L6 推广到逐层，得到向量质量谱 w_l 及其质心 com_V
  （A0/A1/A2 = 26.150 / 26.704 / 26.675 层），并报告邻域 [26,28] 的 **向量** MLP 份额
  share_mlp_nb = 0.740 / 0.975 / 0.824（P5 MLP_DOMINANT_ALL）。
  但这一切都是**几何量**（对"供体-受体模块输出差向量"取范数），不是**行为量**。
  P17 §7 的 H11 明确留下缺口：
    「若行为层同样 MLP 主导 => 升级为因果；若以 attn 为主 => P17 向量份额是几何假象」。

本 Phase 的唯一改动：**把同一批向量的"范数"换成"行为效应"**。
  即：注入 P_{U_l}(Delta_inc,l)、P_{U_l}(Delta_mlp,l)、P_{U_l}(Delta_attn,l)、
  P_{U_l}(Delta_top1,l)、以及 P_{U_l}(d_l)（累积差，P16 的对象），
  读 dDonor = score_of(patched, ds, sid_d) - BASE[rw].sd0（与 Phase 8 的 T 臂逐字同口径）。

**关键设计点（必须写进 seal）**：
  (A) 注入的是**单层**的增量写入 Delta_inc,l（不是累积差），因为 w_l 量的就是它 —— 这才与 P17 同对象。
      累积差 d_l = sum_{l'<=l} Delta_inc,l' 另设一条桥接臂 CUM_ALL，用于对齐 P16 的 J/xhalf。
      => 由此可判定 P17 的 P6（spearman(w_l, J_l) < 0）是否为**对象错配**的假象。
  (B) 向量预算的"精确可加"是**定义**（Delta_inc := Delta_attn + Delta_mlp）；行为量**不**可加，
      故另设**线性残差** r_lin,l = |b_inc - (b_mlp + b_attn)| / max(|b_inc|, eps) 作为非线性诊断。
  (C) 份额类判据用**同口径比**：share_mlp_beh = sum_l |b_mlp,l| / (sum_l |b_mlp,l| + sum_l |b_attn,l|)
      （与 P17 的 share_mlp_nb = sum ||P(Delta_mlp)|| / sum ||P(Delta_inc)|| 结构对应）。

本探针要回答的四件事（seal 冻结前）：
  (1) 单层注入 P_{U_l}(Delta_inc,l) 在**深端**是否仍可测（dDonor 非 0、非 NaN）？
  (2) 组件是否**可分**：b_mlp 与 b_attn 是否落在不同量级（否则份额无区分力）？
  (3) 注入幅度是否离流形：pert_rel,l = ||P_{U_l}(Delta_c,l)|| / ||h_{l+1}^R||；
      以及线性残差 r_lin 的量级。
  (4) 每前向耗时 -> 总预算。
只跑一臂；只做只读测量；**不设任何判据**（判据在 seal 里冻结）。
用法：PROBE_ARM=A0 python tests/deepseek/Phase18/probe_feasibility_phase18.py
"""
import os
import sys
import io
import json
import time
import hashlib

import numpy as np
import torch

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P18T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase18')
P17T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase17')
P16T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase16')
os.makedirs(P18T, exist_ok=True)

TAG = os.environ.get('PROBE_ARM', 'A0')
NPAIR_PROBE = int(os.environ.get('NPAIR', '4'))
SITES_ENV = os.environ.get('SITES', '')
LOG = []


def w(s=''):
    LOG.append(str(s))
    print(s)
    sys.stdout.flush()


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


# --- 材料来源：P16 exec（模板/实例/配对/量化）+ P16 result（REACH 锚）+ P17 exec（同口径确认）
EX = json.load(io.open(os.path.join(P16T, 'execution_phase16.json'), encoding='utf-8'))
R16 = json.load(io.open(os.path.join(P16T, 'result_phase16.json'), encoding='utf-8'))
E17 = json.load(io.open(os.path.join(P17T, 'execution_phase17.json'), encoding='utf-8'))
R17 = json.load(io.open(os.path.join(P17T, 'result_phase17.json'), encoding='utf-8'))

TMPL = EX['template']
SUPS = list(EX['classes'])
INST_ALL = [tuple(x) for x in EX['instances_all']]
DISC = [tuple(x) for x in EX['discovery']]
CONF = [tuple(x) for x in EX['confirmation']]
PAIRS_ALL = [tuple(x) for x in EX['pairs_all']]
DISC_WORDS = set(x[0] for x in DISC)
QUANT = EX['quant']
ARMS = EX['arms']
ARM_ID = [a for a in ARMS if a.startswith(TAG)][0]
acfg = ARMS[ARM_ID]
MDIR = os.path.join(ROOT, 'models', 'hf', acfg['dir'])

# REACH / 质心锚 —— 从 P16 result 读（与 P17 同一来源）
REACH = [int(x) for x in R16['E7_reach'][ARM_ID]['reach']]
ELL_REACH = R16['E7_reach'][ARM_ID]['ell_reach']
LSTAR = R16['E3_localize'][ARM_ID]['L_star_own']
COM_X = R16['E5_concentration'][ARM_ID]['new_stat']['x']['obs_com']
COM_J = R16['E5_concentration'][ARM_ID]['new_stat']['j']['obs_com']
# P17 的 com_V（严格来源：P17 result）
P17V = R17['arms'][ARM_ID]['E5_com_V']

w('=== Phase18 可行性探针 (arm=%s model=%s) ===' % (ARM_ID, acfg['model']))
w('P16 锚: L*_own=%s ell_reach=%s com_layer(x)=%.3f com_layer(J)=%.3f'
  % (LSTAR, ELL_REACH, COM_X, COM_J))
w('P17 锚: com_V=%.4f 邻域=%s share_mlp_nb=%.4f'
  % (P17V['com_V'], P17V['neighbourhood'], P17V['share_mlp_nb']))
w('REACH n=%d' % len(REACH))
w('sha8: P16.result=%s P16.exec=%s P17.result=%s P17.exec=%s'
  % (sha8(os.path.join(P16T, 'result_phase16.json')),
     sha8(os.path.join(P16T, 'execution_phase16.json')),
     sha8(os.path.join(P17T, 'result_phase17.json')),
     sha8(os.path.join(P17T, 'execution_phase17.json'))))

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def attn_out_proj(layer):
    a = layer.self_attn
    for nm in ('o_proj', 'dense', 'out_proj'):
        if hasattr(a, nm):
            return getattr(a, nm), nm
    raise RuntimeError('no attn out proj')


max_mem = {int(k) if str(k).isdigit() else k: v for k, v in QUANT['max_memory'].items()}
bnb = BitsAndBytesConfig(load_in_4bit=True,
                         bnb_4bit_quant_type=QUANT['bnb_4bit_quant_type'],
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=bool(QUANT['bnb_4bit_use_double_quant']))
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MDIR, quantization_config=bnb, trust_remote_code=True,
    attn_implementation=QUANT['attn_implementation'], low_cpu_mem_usage=True,
    device_map=QUANT['device_map'], max_memory=max_mem)
model.eval()
w('loaded %.1fs' % (time.time() - t0))

layers = model.model.layers
L = len(layers)
CFG = model.config
HID = int(CFG.hidden_size)
NH = int(CFG.num_attention_heads)
HDP = int(getattr(CFG, 'head_dim', HID // max(NH, 1)))
OPR = [attn_out_proj(layers[l])[0] for l in range(L)]
MLPS = [layers[l].mlp for l in range(L)]
OIN = int(OPR[0].in_features)
w('cfg L=%d HID=%d NH=%d head_dim=%d o_proj_in=%d(%s) tied=%s'
  % (L, HID, NH, HDP, OIN, attn_out_proj(layers[0])[1],
     bool(getattr(CFG, 'tie_word_embeddings', False))))
assert OIN == NH * HDP, 'F4: o_proj_in %d != NH*HD %d' % (OIN, NH * HDP)


def ids_of(s):
    return tok.encode(s, add_special_tokens=False)


# --- F1b 类别 token 逐臂现场解析（Phase 15 amend1 修复条款）
SUP_ID = {}
for _wd in SUPS:
    _t = list(tok.encode(_wd, add_special_tokens=False))
    assert len(_t) == 1 and tok.decode([_t[0]]) == _wd, 'F1b 失败: %r %r' % (_wd, _t)
    SUP_ID[_wd] = int(_t[0])
w('F1b 类别 token: %s' % json.dumps(SUP_ID, ensure_ascii=False))

# --- 前向计时
ii0 = torch.tensor([ids_of(TMPL % '苹果')], device='cuda')
with torch.no_grad():
    for _ in range(3):
        model(input_ids=ii0)
    t_f = time.time()
    N_T = 10
    for _ in range(N_T):
        model(input_ids=ii0)
    fwd_s = (time.time() - t_f) / N_T
w('forward %.4f s' % fwd_s)


# --- 分块掩码前向（模块自身前向；nf4 下不能直接乘打包权重）
_NH1 = NH + 1
_blocks = np.zeros((_NH1, OIN), np.float32)


@torch.no_grad()
def head_blocks(opmod, v):
    _blocks[:] = 0.0
    for h in range(NH):
        _blocks[h, h * HDP:(h + 1) * HDP] = v[h * HDP:(h + 1) * HDP]
    _blocks[NH] = v
    t = torch.tensor(_blocks, device='cuda', dtype=torch.bfloat16)
    o = opmod(t).float().cpu().numpy()
    return o[:NH], o[NH]


# --- 扩展 capture（O = o_proj 输入；M = MLP 输出）
@torch.no_grad()
def capture(text):
    ii = torch.tensor([ids_of(text)], device='cuda')
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
cap_s = time.time() - t0
w('capture %d instances in %.1fs' % (len(CAP), cap_s))
wd0 = INST_ALL[0][0]
assert not np.isnan(CAP[wd0][0]).any()

# --- 保真度门
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
    for l in (3, 6, 12, 24, 33):
        hb, full = head_blocks(OPR[l], O[l])
        blk.append(float(np.linalg.norm(hb.sum(0) - full)) / max(float(np.linalg.norm(full)), 1e-9))
arch = np.array(arch); blk = np.array(blk)
w('FIDELITY arch max=%.3e mean=%.3e | blocks max=%.3e mean=%.3e'
  % (arch.max(), arch.mean(), blk.max(), blk.mean()))

# --- 类子空间 U_l（与 P16/P17 同口径：全实例按类平均）
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
w('U_ell: rank=%d n_classes=%d' % (rk, len(AVAIL)))


def proj(v, Ub):
    return (v @ Ub.T) @ Ub


# --- BASE（与 Phase 8 T 臂 / P16 逐字同口径）
def score_of(v, sup, sid):
    v = v.copy(); v[sid] = -1e9
    own = float(v[SUP_ID[sup]])
    others = [float(v[SUP_ID[x]]) for x in SUPS if x != sup]
    return own - float(np.mean(others))


PAIRS = [p for p in PAIRS_ALL if p[0] in DISC_WORDS and p[0] in CAP and p[2] in CAP][:NPAIR_PROBE]
BASE = {}
for (rw, rs, dw, ds, sw) in PAIRS:
    lg = CAP[rw][3]
    BASE[rw] = dict(sr0=score_of(lg, rs, ids_of(rw)[0]), sd0=score_of(lg, ds, ids_of(dw)[0]))
bad = [rw for rw, b in BASE.items() if not (b['sr0'] > 0)]
w('F2 base: n=%d 受体类 mean=%+.3f 供体类 mean=%+.3f bad=%s'
  % (len(BASE), float(np.mean([b['sr0'] for b in BASE.values()])),
     float(np.mean([b['sd0'] for b in BASE.values()])), bad if bad else 'NONE'))


# --- 单点注入
@torch.no_grad()
def fwd_patch(text, site, vec):
    ii = torch.tensor([ids_of(text)], device='cuda')

    def hook(mod, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        t = t.clone()
        t[0, -1, :] = torch.as_tensor(vec, device=t.device, dtype=t.dtype)
        return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t
    h = layers[site].register_forward_hook(hook)
    try:
        o = model(input_ids=ii)
    finally:
        h.remove()
    return o.logits[0, -1].float().detach().cpu().numpy()


# --- 站点选择
if SITES_ENV:
    SITES = [int(x) for x in SITES_ENV.split(',')]
else:
    SITES = [6, 14, 26, 34]
SITES = [s for s in SITES if 1 <= s <= L - 1]
w('probe sites = %s (n=%d, 域 1..L-1)' % (SITES, len(SITES)))

ARMS_C = [('INC_ALL', 1.0), ('INC_ALL', 0.5), ('INC_MLP', 1.0),
          ('INC_ATTN', 1.0), ('INC_TOP1', 1.0), ('CUM_ALL', 1.0)]

n_fw = 0
out_rows = {}
t0 = time.time()
for s in SITES:
    row = {}
    for (cname, alpha) in ARMS_C:
        dd = 0.0
        rel, nrm = [], []
        per = []
        lin_res = []
        for (rw, rs, dw, ds, sw) in PAIRS:
            HR = CAP[rw][0]; HD = CAP[dw][0]
            OR_, MR = CAP[rw][1], CAP[rw][2]
            OD, MD = CAP[dw][1], CAP[dw][2]
            Ub = U[s]
            hbD, fullD = head_blocks(OPR[s], OD[s])
            hbR, fullR = head_blocks(OPR[s], OR_[s])
            d_attn = fullD - fullR
            d_mlp = MD[s] - MR[s]
            d_inc = d_attn + d_mlp
            dd_h = hbD - hbR                    # [NH, HID]
            d_cum = HD[s + 1] - HR[s + 1]       # 累积差（P16 的对象）
            if cname == 'INC_ALL':
                dv = d_inc
            elif cname == 'INC_MLP':
                dv = d_mlp
            elif cname == 'INC_ATTN':
                dv = d_attn
            elif cname == 'INC_TOP1':
                pv = np.array([float(np.linalg.norm(proj(dd_h[h], Ub))) for h in range(NH)])
                dv = dd_h[int(np.argmax(pv))]
            elif cname == 'CUM_ALL':
                dv = d_cum
            else:
                raise RuntimeError(cname)
            pv = proj(dv.astype(np.float32), Ub) * float(alpha)
            h0 = HR[s + 1].astype(np.float32)
            lg = fwd_patch(TMPL % rw, s, h0 + pv)
            n_fw += 1
            x1 = score_of(lg, ds, ids_of(dw)[0]) - BASE[rw]['sd0']
            dd += x1; per.append(float(x1))
            nrm.append(float(np.linalg.norm(pv)))
            rel.append(float(np.linalg.norm(pv)) / max(float(np.linalg.norm(h0)), 1e-9))
            # 线性残差改为**离线**从 INC_MLP / INC_ATTN 两条读数计算，避免重复前向
            # （勘误 P-B：原实现每条 ALL 前向后再补 2 条 MLP/ATTN 前向，纯属重复）。
        n = max(len(PAIRS), 1)
        row['%s@a%.2f' % (cname, alpha)] = dict(
            dDonor=dd / n, n=n, pert_rel_mean=float(np.mean(rel)),
            pv_norm_mean=float(np.mean(nrm)), per_pair=per,
            lin_res_mean=(float(np.mean(lin_res)) if lin_res else None))
    out_rows[str(s)] = row
    w('--- L%-3d' % s)
    for k, v in row.items():
        w('     %-12s dD=%+8.3f pert_rel=%.4f |pv|=%8.3f lin=%s'
          % (k, v['dDonor'], v['pert_rel_mean'], v['pv_norm_mean'],
             ('%.3f' % v['lin_res_mean']) if v['lin_res_mean'] is not None else 'NA'))
el_s = time.time() - t0
w('sweep done %.1fs ; forwards=%d ; per-forward=%.4f s' % (el_s, n_fw, el_s / max(n_fw, 1)))

# --- 组件可分性摘要（把所有位点合起来看量级）
def _agg(key):
    vals = []
    for s in SITES:
        r = out_rows[str(s)].get(key)
        if r:
            vals.append(r['dDonor'])
    return vals


for key in ('INC_ALL@a1.00', 'INC_MLP@a1.00', 'INC_ATTN@a1.00', 'INC_TOP1@a1.00', 'CUM_ALL@a1.00'):
    vv = _agg(key)
    w('AGG %-12s dD=%s' % (key, ' '.join('%+8.3f' % x for x in vv)))

out = dict(arm=ARM_ID, model=acfg['model'], L=L, HID=HID, NH=NH, head_dim=HDP, o_proj_in=OIN,
           fwd_s=fwd_s, capture_s=cap_s, sweep_s=el_s, n_forwards=n_fw, n_pairs=len(PAIRS),
           sites=SITES, reach=REACH, ell_reach=ELL_REACH, L_star_own=LSTAR,
           fidelity_arch=dict(max=float(arch.max()), mean=float(arch.mean())),
           fidelity_blocks=dict(max=float(blk.max()), mean=float(blk.mean())),
           f2_base_bad=bad,
           p16_com_layer_x=COM_X, p16_com_layer_j=COM_J, p17_com_V=P17V['com_V'],
           p17_neighbourhood=P17V['neighbourhood'], p17_share_mlp_nb=P17V['share_mlp_nb'],
           rows=out_rows,
           sha8=dict(p16_result=sha8(os.path.join(P16T, 'result_phase16.json')),
                     p16_exec=sha8(os.path.join(P16T, 'execution_phase16.json')),
                     p17_result=sha8(os.path.join(P17T, 'result_phase17.json')),
                     p17_exec=sha8(os.path.join(P17T, 'execution_phase17.json'))),
           arm_order=E17['arm_order'])
json.dump(out, io.open(os.path.join(P18T, '_probe_feasibility_%s.json' % TAG), 'w',
                       encoding='utf-8'), ensure_ascii=False, indent=1)
io.open(os.path.join(P18T, '_probe_feasibility_%s.txt' % TAG), 'w', encoding='utf-8').write(
    '\r\n'.join(LOG) + '\r\n')
w('WROTE _probe_feasibility_%s.{json,txt}' % TAG)
