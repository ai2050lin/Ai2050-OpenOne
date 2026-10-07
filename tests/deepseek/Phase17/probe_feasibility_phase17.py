# -*- coding: utf-8 -*-
"""
Phase 17 (N2h1-alpha-10) 可行性探针 —— 「位置 -> 组件」写入向量质心。

设计问题（seal 冻结前必须验证的四件事）：
  (1) o_proj 输入 / MLP 输出钩子在架构上都能取到，形状与 F4 断言一致；
  (2) 分块分解的**保真度**：按头掩码经**模块自身前向**得到的 sum_h Delta_head 与
      o_proj(v) 的残差；以及架构恒等式 (h_{l+1}-h_l) == attn_out_l + mlp_out_l 的数值精度。
      （注意：向量预算的"精确可加"是**定义**——Delta_inc := Delta_attn + Delta_mlp；
        恒等式检查是**保真度门**，容差由本探针实测的量化噪声地板决定。）
  (3) 向量质量谱 w_l = mean_pairs ||P_{U_l}(Delta_l)|| 的量级与形状；质心 com_V 落在哪；
  (4) 每前向耗时 -> 总预算。

只跑一臂；只做只读测量；**不设任何判据**（判据在 seal 里冻结）。
用法：PROBE_ARM=A0 python tests/deepseek/Phase17/probe_feasibility_phase17.py
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
P17T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase17')
P16T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase16')
os.makedirs(P17T, exist_ok=True)

TAG = os.environ.get('PROBE_ARM', 'A0')
NPAIR_PROBE = int(os.environ.get('NPAIR', '24'))
LOG = []


def w(s=''):
    LOG.append(str(s))
    print(s)
    sys.stdout.flush()


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


EX = json.load(io.open(os.path.join(P16T, 'execution_phase16.json'), encoding='utf-8'))
R16 = json.load(io.open(os.path.join(P16T, 'result_phase16.json'), encoding='utf-8'))
TMPL = EX['template']
SUPS = list(EX['classes'])
INST_ALL = [tuple(x) for x in EX['instances_all']]
DISC = [tuple(x) for x in EX['discovery']]          # [实例词, 类]
PAIRS_ALL = [tuple(x) for x in EX['pairs_all']]      # (rw, rs, dw, ds, sw)
DISC_WORDS = set(x[0] for x in DISC)
QUANT = EX['quant']
ARMS = EX['arms']
ARM_ID = [a for a in ARMS if a.startswith(TAG)][0]
acfg = ARMS[ARM_ID]
MDIR = os.path.join(ROOT, 'models', 'hf', acfg['dir'])
REACH = [int(x) for x in R16['E7_reach'][ARM_ID]['reach']]
ELL_REACH = R16['E7_reach'][ARM_ID]['ell_reach']
COM_X = R16['E5_concentration'][ARM_ID]['new_stat']['x']['obs_com']
COM_J = R16['E5_concentration'][ARM_ID]['new_stat']['j']['obs_com']
LSTAR = R16['E3_localize'][ARM_ID]['L_star_own']

w('=== Phase17 可行性探针 (arm=%s model=%s) ===' % (ARM_ID, acfg['model']))
w('anchors from P16: L*_own=%s ell_reach=%s com_layer(x)=%.3f com_layer(J)=%.3f'
  % (LSTAR, ELL_REACH, COM_X, COM_J))
w('REACH n=%d' % len(REACH))
w('P16 result sha8=%s exec sha8=%s'
  % (sha8(os.path.join(P16T, 'result_phase16.json')),
     sha8(os.path.join(P16T, 'execution_phase16.json'))))

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


# --- 分块掩码前向：一次调用得到 [head0..head_{NH-1}, full]
@torch.no_grad()
def head_blocks(opmod, v):
    """v: (OIN,) float32 numpy -> (heads (NH,HID) float32, full (HID,) float32)"""
    M = np.zeros((NH + 1, OIN), np.float32)
    for h in range(NH):
        M[h, h * HDP:(h + 1) * HDP] = v[h * HDP:(h + 1) * HDP]
    M[NH] = v
    t = torch.tensor(M, device='cuda', dtype=torch.bfloat16)
    o = opmod(t)
    o = o.float().cpu().numpy()
    return o[:NH], o[NH]


# --- 扩展 capture
@torch.no_grad()
def capture(text):
    ii = torch.tensor([ids_of(text)], device='cuda')
    store_o, store_m = {}, {}
    hs = []
    for l in range(L):
        def mk_o(l):
            def f(mod, args):
                store_o[l] = args[0].detach()[0, -1, :].float().cpu().numpy().copy()
            return f

        def mk_m(l):
            def f(mod, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                store_m[l] = t.detach()[0, -1, :].float().cpu().numpy().copy()
            return f
        hs.append(OPR[l].register_forward_pre_hook(mk_o(l)))
        hs.append(MLPS[l].register_forward_hook(mk_m(l)))
    out = model(input_ids=ii, output_hidden_states=True)
    for h in hs:
        h.remove()
    HH = np.stack([x[0, -1].float().detach().cpu().numpy() for x in out.hidden_states], 0)
    return HH, store_o, store_m, out.logits[0, -1].float().detach().cpu().numpy()


t0 = time.time()
CAP = {}
for wd, sup in INST_ALL:
    CAP[wd] = capture(TMPL % wd)
cap_s = time.time() - t0
w('capture %d instances in %.1fs (%.4f s each)' % (len(CAP), cap_s, cap_s / len(CAP)))
wd0 = INST_ALL[0][0]
w('shapes: HH%s O_l%d%s M_l%d%s' % (CAP[wd0][0].shape, L - 1, CAP[wd0][1][L - 1].shape,
                                    L - 1, CAP[wd0][2][L - 1].shape))
assert not np.isnan(CAP[wd0][0]).any()

# --- (2) 保真度：架构恒等式 + 分块可加性
arch, part = [], []
for wd, sup in INST_ALL[:6]:
    HH, O, M, _ = CAP[wd]
    for l in range(L - 1):
        lhs = HH[l + 1] - HH[l]
        rhs = OPR[l](torch.tensor(O[l][None, :], device='cuda', dtype=torch.bfloat16)
                     ).float().cpu().numpy()[0] + M[l]
        arch.append(float(np.linalg.norm(lhs - rhs)) / max(float(np.linalg.norm(lhs)), 1e-9))
for wd, sup in INST_ALL[:4]:
    HH, O, M, _ = CAP[wd]
    for l in (3, 6, 12, 24, 33):
        hb, full = head_blocks(OPR[l], O[l])
        part.append(float(np.linalg.norm(hb.sum(0) - full)) / max(float(np.linalg.norm(full)), 1e-9))
arch = np.array(arch); part = np.array(part)
w('FIDELITY arch  ||(h_{l+1}-h_l) - (attn_out_l + m_l)||/||.| : max=%.3e mean=%.3e p99=%.3e'
  % (arch.max(), arch.mean(), float(np.percentile(arch, 99))))
w('FIDELITY blocks ||sum_h head_block - o_proj(v)||/||o_proj(v)|| : max=%.3e mean=%.3e'
  % (part.max(), part.mean()))

# --- 类子空间 U_l（level=l+1，与 P16 E3 同口径）
by = {}
for wd, sup in INST_ALL:
    by.setdefault(sup, []).append(wd)
AVAIL = [s for s in SUPS if s in by]
r = max(len(AVAIL) - 1, 1)
U = {}
SING6 = None
for l in range(L):
    mus = np.stack([np.mean([CAP[wd][0][l + 1] for wd in by[s]], 0) for s in AVAIL], 0).astype(np.float64)
    D = mus - mus.mean(0, keepdims=True)
    _, sv, Vt = np.linalg.svd(D, full_matrices=False)
    U[l] = Vt[:r].astype(np.float32)
    if l == 6:
        SING6 = sv[:r]
w('U_ell: rank=%d n_classes=%d ; sing@L6=%s'
  % (r, len(AVAIL), ' '.join('%.2f' % x for x in SING6)))


def proj(v, Ub):
    return (v @ Ub.T) @ Ub


def centroid_on_grid(mass, sites):
    """逐层质量按 sites 区间聚合，mid=(s_j+s_{j+1})/2（与 stat_com_layer 同一 mid 定义）。"""
    tot = 0.0
    num = 0.0
    for j in range(len(sites) - 1):
        lo, hi = sites[j], sites[j + 1]
        s = float(sum(mass[l] for l in range(lo, hi) if l < len(mass)))
        tot += s
        num += s * ((lo + hi) / 2.0)
    return (num / tot) if tot > 1e-12 else None


# --- (3) 向量质量谱
PAIRS = [p for p in PAIRS_ALL if p[0] in DISC_WORDS and p[0] in CAP and p[2] in CAP][:NPAIR_PROBE]
w('pairing: usable discovery pairs used = %d' % len(PAIRS))
t0 = time.time()
wb_all = np.zeros(L - 1); wb_attn = np.zeros(L - 1); wb_mlp = np.zeros(L - 1); wb_top = np.zeros(L - 1)
head_mass = {h: np.zeros(L - 1) for h in range(NH)}
for (rw, rs, dw, ds, sw) in PAIRS:
    hR, OR_, MR, _ = CAP[rw]
    hD, OD, MD, _ = CAP[dw]
    for l in range(L - 1):
        Ub = U[l]
        hbD, fullD = head_blocks(OPR[l], OD[l])
        hbR, fullR = head_blocks(OPR[l], OR_[l])
        d_attn = fullD - fullR
        d_mlp = MD[l] - MR[l]
        wb_all[l] += float(np.linalg.norm(proj(d_attn + d_mlp, Ub)))
        wb_attn[l] += float(np.linalg.norm(proj(d_attn, Ub)))
        wb_mlp[l] += float(np.linalg.norm(proj(d_mlp, Ub)))
        dd = hbD - hbR
        per = np.array([float(np.linalg.norm(proj(dd[h], Ub))) for h in range(NH)])
        wb_top[l] += float(per.max())
        for h in range(NH):
            head_mass[h][l] += per[h]
decomp_s = time.time() - t0
npair = len(PAIRS)
for arr in (wb_all, wb_attn, wb_mlp, wb_top):
    arr /= npair
for h in head_mass:
    head_mass[h] /= npair
w('mass profile decomposed in %.1fs (%.4f s per pair-layer)'
  % (decomp_s, decomp_s / max(npair * (L - 1), 1)))

RE = [s for s in REACH if s <= L - 1]
comV = centroid_on_grid(wb_all, RE)
comV_mlp = centroid_on_grid(wb_mlp, RE)
comV_attn = centroid_on_grid(wb_attn, RE)
comV_top = centroid_on_grid(wb_top, RE)
comV_full = centroid_on_grid(wb_all, list(range(1, L)))
w('')
w('--- 向量质量谱 w_l ---')
for l in range(L - 1):
    tg = '*' if l in RE else ' '
    w('  %s L%-3d all=%9.3f attn=%9.3f mlp=%9.3f top1=%9.3f  mlp_share=%.3f'
      % (tg, l, wb_all[l], wb_attn[l], wb_mlp[l], wb_top[l], wb_mlp[l] / max(wb_all[l], 1e-9)))
w('')
w('com_V(all)  = %s  [REACH 域]' % ('%.3f' % comV if comV else None))
w('com_V(mlp)  = %s ; com_V(attn) = %s ; com_V(top1head) = %s'
  % (('%.3f' % comV_mlp if comV_mlp else None), ('%.3f' % comV_attn if comV_attn else None),
     ('%.3f' % comV_top if comV_top else None)))
w('com_V(all)  = %s  [全域 1..L-1]' % ('%.3f' % comV_full if comV_full else None))
w('P16 锚: L*_own=%s ell_reach=%s com_layer(x)=%.3f com_layer(J)=%.3f' % (LSTAR, ELL_REACH, COM_X, COM_J))
tot_h = {h: float(head_mass[h].sum()) for h in head_mass}
top5 = sorted(tot_h, key=lambda z: -tot_h[z])[:5]
w('--- 逐头质量前 5 ---')
for h in top5:
    w('  head%-3d total=%9.3f com_V=%s' % (h, tot_h[h], centroid_on_grid(head_mass[h], RE)))
w('--- 组件份额（REACH 域总量） ---')
tot_all = float(wb_all[RE].sum()); tot_a = float(wb_attn[RE].sum()); tot_m = float(wb_mlp[RE].sum())
w('  attn=%.4f  mlp=%.4f  (attn+mlp)/all=%.6f' % (tot_a / tot_all, tot_m / tot_all, (tot_a + tot_m) / tot_all))

out = dict(arm=ARM_ID, model=acfg['model'], L=L, HID=HID, NH=NH, head_dim=HDP, o_proj_in=OIN,
           fwd_s=fwd_s, capture_s=cap_s, decomp_s=decomp_s, n_inst=len(CAP), n_pairs=npair,
           fidelity_arch=dict(max=float(arch.max()), mean=float(arch.mean()),
                              p99=float(np.percentile(arch, 99))),
           fidelity_blocks=dict(max=float(part.max()), mean=float(part.mean())),
           reach=RE, ell_reach=ELL_REACH, L_star_own=LSTAR,
           com_layer_p16_x=COM_X, com_layer_p16_j=COM_J,
           com_V=comV, com_V_mlp=comV_mlp, com_V_attn=comV_attn, com_V_top=comV_top,
           com_V_full=comV_full, w_all=[float(x) for x in wb_all], w_attn=[float(x) for x in wb_attn],
           w_mlp=[float(x) for x in wb_mlp], w_top=[float(x) for x in wb_top],
           head_mass={str(h): float(tot_h[h]) for h in tot_h},
           head_comv={str(h): centroid_on_grid(head_mass[h], RE) for h in tot_h},
           head_top5=[int(h) for h in top5],
           share_attn=tot_a / tot_all, share_mlp=tot_m / tot_all,
           p16_result_sha8=sha8(os.path.join(P16T, 'result_phase16.json')))
json.dump(out, io.open(os.path.join(P17T, '_probe_feasibility_%s.json' % TAG), 'w',
                       encoding='utf-8'), ensure_ascii=False, indent=1)
io.open(os.path.join(P17T, '_probe_feasibility_%s.txt' % TAG), 'w', encoding='utf-8').write(
    '\r\n'.join(LOG) + '\r\n')
w('WROTE _probe_feasibility_%s.{json,txt}' % TAG)
