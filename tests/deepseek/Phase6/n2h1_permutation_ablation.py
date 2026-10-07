# -*- coding: utf-8 -*-
"""
N2-h1: 置换向量消融 —— 承诺层是否由"类别分量"承载？
=====================================================
动机：N2c 用【整块 hidden state】移植得到 L6 一次性锁死（+10.11）。但整块移植同时注入了
      范数、位置、身份指纹与类别内容 —— 无法区分"内容注入"与"几何破坏"。
      本探针把移植分解为【类别子空间分量】与【正交残差】两部分，并配随机子空间地板。

设计（预注册见 N2h1_design_seal.json）：
  U_l = span{类均值方向}（6 类去全局均值 -> rank 5），per-layer 估计
  A_full  h <- h_donor
  B_cat   h <- h_rec + P_U (h_donor - h_rec)
  C_resid h <- h_rec + (I-P_U)(h_donor - h_rec)
  D_rand  h <- h_rec + P_V (h_donor - h_rec)     V = 随机 5 维正交基
  E_same  donor 取同类实例，B 模式（类别不变对照）
  S_self  donor = recipient 自身（sanity，应 ~0）
"""
import os, sys, time, json
import numpy as np
import torch

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'qwen3-4b'
MDIR = os.path.join(ROOT, 'models', 'hf', MODEL)
OUT = os.path.join(ROOT, 'tests', 'gpt5_temp', 'n2h1_report_%s.txt' % MODEL.replace('/', '_'))
lines = []
def w(s=''):
    lines.append(str(s)); print(s); sys.stdout.flush()

from transformers import AutoTokenizer, AutoModelForCausalLM
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MDIR, dtype=torch.bfloat16,
                                            trust_remote_code=True).to('cuda').eval()
_core = getattr(model.model, 'language_model', model.model)
layers = _core.layers; L = len(layers); norm = _core.norm; head = model.lm_head
HID = model.config.hidden_size
def ids_of(s): return tok.encode(s, add_special_tokens=False)

RNG = np.random.default_rng(20261001)

GROUPS = {
    '水果': ['苹果', '香蕉', '梨', '西瓜', '葡萄', '草莓', '芒果', '柠檬'],
    '动物': ['狗', '猫', '老虎', '大象', '兔子', '猴子', '马', '牛'],
    '交通工具': ['汽车', '火车', '飞机', '摩托车', '卡车', '地铁'],
    '家具': ['桌子', '椅子', '床', '沙发', '地毯', '窗帘'],
    '金属': ['铁', '铜', '铝', '金', '银', '锌', '铅'],
    '颜色': ['红', '蓝', '绿', '黄', '黑', '白'],
}
SUP_ID = {s: ids_of(s)[0] for s in GROUPS if len(ids_of(s)) == 1}
SUPS = list(SUP_ID.keys())
INST = [(wd, sup) for sup, ms in GROUPS.items() if sup in SUP_ID
        for wd in ms if len(ids_of(wd)) == 1]

PATCH_L = [1, 3, 5, 6, 7, 9, 12, 15, 17, 20, 24, 27, 30, 33, 34, 35]
PATCH_L = [l for l in PATCH_L if l < L]
RANKS = [1, 2, 3, 5, 10, 20, 50]
TMPL = '%s是一种'

w('=== N2-h1 置换向量消融 ===')
w('seal N2h1_design_seal.json ; time %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
w('model=%s L=%d hid=%d classes=%d instances=%d tie=%s' %
  (MODEL, L, HID, len(SUPS), len(INST), model.config.tie_word_embeddings))
w('patch layers=%s' % PATCH_L)
w('instances: ' + ' '.join('%s(%s)' % (a, b) for a, b in INST))
sys.stdout.flush()

def score_rank(v, sup, sid):
    v = v.copy(); v[sid] = -1e9
    own = v[SUP_ID[sup]]
    others = [v[SUP_ID[x]] for x in SUPS if x != sup]
    order = np.argsort(-v)
    return float(own - np.mean(others)), int(np.where(order == SUP_ID[sup])[0][0]) + 1

@torch.no_grad()
def fwd(text, patch=None):
    ii = torch.tensor([ids_of(text)], device='cuda')
    hs = []
    if patch is not None:
        l, newvec = patch
        def hook(mod, args, out):
            if isinstance(out, tuple):
                h = out[0].clone(); h[0, -1, :] = newvec.to(h.dtype)
                return (h,) + tuple(out[1:])
            h = out.clone(); h[0, -1, :] = newvec.to(h.dtype); return h
        hs.append(layers[l].register_forward_hook(hook))
    out = model(input_ids=ii)
    for h in hs: h.remove()
    return out.logits[0, -1].float().detach().cpu().numpy()

@torch.no_grad()
def hidden_all(text):
    ii = torch.tensor([ids_of(text)], device='cuda')
    out = model(input_ids=ii, output_hidden_states=True)
    return np.stack([h[0, -1].float().detach().cpu().numpy() for h in out.hidden_states], 0)

# ---------------- 0. 采集状态 & 估计类子空间 ----------------
HB = {}
for wd, sup in INST:
    HB[wd] = hidden_all(TMPL % wd)
w('hidden states collected for %d instances' % len(HB))
sys.stdout.flush()

UB = {}     # layer l -> [5, HID] orthonormal basis of class subspace
VB = {}     # layer l -> [5, HID] random orthonormal basis
by_class = {}
for wd, sup in INST:
    by_class.setdefault(sup, []).append(wd)
for l in PATCH_L:
    mus = np.stack([np.mean([HB[wd][l + 1] for wd in by_class[s]], 0) for s in SUPS], 0).astype(np.float64)
    gm = mus.mean(0, keepdims=True)
    D = mus - gm
    _, S, Vt = np.linalg.svd(D, full_matrices=False)
    k = min(len(SUPS) - 1, 5)
    UB[l] = Vt[:k].astype(np.float32)
    rq, _ = np.linalg.qr(RNG.standard_normal((k, HID)).astype(np.float32).T)
    VB[l] = rq.T.astype(np.float32)
w('class subspace rank=%d ; singular values(L%d)=%s' %
  (UB[PATCH_L[0]].shape[0], PATCH_L[0], ' '.join('%.1f' % x for x in S[:6])))
sys.stdout.flush()

def proj(vec, U):
    return (vec @ U.T) @ U

# ---------------- 1. pair 构造 ----------------
IDX = {wd: i for i, (wd, _) in enumerate(INST)}
PAIRS = []
n = len(INST)
for i, (rw, rs) in enumerate(INST):
    j = (i + n // 2) % n
    while INST[j][1] == rs:
        j = (j + 1) % n
    dw, ds = INST[j]
    same = [x for x, s in INST if s == rs and x != rw]
    sw = same[(i * 3) % len(same)] if same else None
    PAIRS.append((rw, rs, dw, ds, sw, rs))
w('')
w('pairs n=%d  (recipient -> cross-class donor | same-class donor)' % len(PAIRS))
for p in PAIRS[:6]:
    w('  %s(%s) <- %s(%s) | same=%s' % (p[0], p[1], p[2], p[3], p[4]))
sys.stdout.flush()

# ---------------- 2. base ----------------
BASE = {}
for rw, rs, dw, ds, sw, ss in PAIRS:
    pr = TMPL % rw
    sid_r = ids_of(rw)[0]; sid_d = ids_of(dw)[0]
    v0 = fwd(pr)
    sr0, rr0 = score_rank(v0, rs, sid_r)
    sd0, rd0 = score_rank(v0, ds, sid_d)
    BASE[rw] = dict(pr=pr, sid_r=sid_r, sid_d=sid_d, sr0=sr0, sd0=sd0, rr0=rr0, rd0=rd0)
w('')
w('base: recipient 类分数 mean=%.3f ; donor 类分数 mean=%.3f (应远低于受体类分数)' %
  (np.mean([BASE[r]['sr0'] for r, *_ in PAIRS]), np.mean([BASE[r]['sd0'] for r, *_ in PAIRS])))
sys.stdout.flush()

def run_mode(mode):
    """返回 {layer: (dRecip, dDonor, n)}"""
    res = {}
    for l in PATCH_L:
        Ub = UB[l]; Vb = VB[l]
        dr_l = []; dd_l = []
        for (rw, rs, dw, ds, sw, ss) in PAIRS:
            B = BASE[rw]
            src = rw if mode == 'S_self' else (sw if mode == 'E_same' else dw)
            h_rec = HB[rw][l + 1].astype(np.float32)
            h_don = HB[src][l + 1].astype(np.float32)
            diff = h_don - h_rec
            if mode == 'A_full':
                new = h_don
            elif mode in ('B_cat', 'E_same'):
                new = h_rec + proj(diff, Ub)
            elif mode == 'C_resid':
                new = h_rec + (diff - proj(diff, Ub))
            elif mode == 'D_rand':
                new = h_rec + proj(diff, Vb)
            elif mode == 'S_self':
                new = h_don
            vec = torch.tensor(new, device='cuda')
            v1 = fwd(B['pr'], patch=(l, vec))
            sr1, _ = score_rank(v1, rs, B['sid_r'])
            sd1, _ = score_rank(v1, ds, B['sid_d'])
            dr_l.append(sr1 - B['sr0']); dd_l.append(sd1 - B['sd0'])
        res[l] = (float(np.mean(dr_l)), float(np.mean(dd_l)), len(dr_l))
    return res

MODES = ['S_self', 'A_full', 'B_cat', 'C_resid', 'D_rand', 'E_same']
CURVE = {}
for m in MODES:
    CURVE[m] = run_mode(m)
    w('  [%s] done @%.0fs' % (m, time.time() - t0)); sys.stdout.flush()

# ---------------- 3. 报告曲线 ----------------
w('')
w('--- 承诺曲线（dDonor = 受体句输出转向 donor 类的分数变化）---')
w('%-5s | %s' % ('L', '  '.join('%9s' % m for m in MODES)))
for l in PATCH_L:
    w('%-5d | %s' % (l, '  '.join('%+9.3f' % CURVE[m][l][1] for m in MODES)))
w('')
w('--- dRecip（受体自身类分数的变化）---')
w('%-5s | %s' % ('L', '  '.join('%9s' % m for m in MODES)))
for l in PATCH_L:
    w('%-5d | %s' % (l, '  '.join('%+9.3f' % CURVE[m][l][0] for m in MODES)))

# ---------------- 4. 判据裁决 ----------------
w('')
w('--- 预注册判据裁决 ---')
A = CURVE['A_full']; B = CURVE['B_cat']; D = CURVE['D_rand']; E = CURVE['E_same']; C = CURVE['C_resid']
lA = PATCH_L[int(np.argmax([A[l][1] for l in PATCH_L]))]
w('A_full  dDonor 峰值 @L%d = %+.3f ; 末层(L%d) = %+.3f' % (lA, A[lA][1], PATCH_L[-1], A[PATCH_L[-1]][1]))
lB = PATCH_L[int(np.argmax([B[l][1] for l in PATCH_L]))]
w('B_cat   dDonor 峰值 @L%d = %+.3f ; 末层 = %+.3f' % (lB, B[lB][1], B[PATCH_L[-1]][1]))
w('D_rand  dDonor 峰值 @L%d = %+.3f ; 末层 = %+.3f' % (
    PATCH_L[int(np.argmax([D[l][1] for l in PATCH_L]))],
    max(D[l][1] for l in PATCH_L), D[PATCH_L[-1]][1]))
w('C_resid 末层 = %+.3f ; E_same 末层 = %+.3f ; S_self 末层 = %+.3f' %
  (C[PATCH_L[-1]][1], E[PATCH_L[-1]][1], CURVE['S_self'][PATCH_L[-1]][1]))
w('')
ratioB = B[PATCH_L[-1]][1] / max(abs(A[PATCH_L[-1]][1]), 1e-9)
w('末层 B/A = %.3f  (H1-A 需 >=0.70 ; H1-B 为 <0.30)' % ratioB)
# 跳变检测（相邻层最大增量）
jumps = [(l, B[l][1] - B[PATCH_L[i - 1]][1]) for i, l in enumerate(PATCH_L) if i > 0]
lj, jv = max(jumps, key=lambda x: x[1]) if jumps else (None, 0)
w('B 最大相邻层增量: L%d (+%.3f)  (L6 前一层值=%s)' %
  (lj, jv, ('%+.3f' % B[5][1]) if 5 in B else 'NA'))
w('E/B = %.3f  (对照需 <0.30 表示同类不变)' % (E[PATCH_L[-1]][1] / max(abs(B[PATCH_L[-1]][1]), 1e-9)))
w('D/B = %.3f  (需 <1.0 表示类别特异，非任意扰动)' % (D[PATCH_L[-1]][1] / max(abs(B[PATCH_L[-1]][1]), 1e-9)))
if ratioB >= 0.70 and jv > 3.0:
    w('>>> 裁决 H1-A：承诺由类别分量承载（内容注入）')
elif ratioB < 0.30 or abs(D[PATCH_L[-1]][1]) >= abs(B[PATCH_L[-1]][1]):
    w('>>> 裁决 H1-B：承诺是几何伪影（N2 系列关闭）')
else:
    w('>>> 裁决 H1-C：混合（内容与几何各贡献一部分）')
sys.stdout.flush()

# ---------------- 5. rank sweep（维数敏感性）----------------
w('')
w('--- rank sweep @承诺层 L%d（B 模式，看类别子空间维数的作用）---' % lB)
mus_sw = np.stack([np.mean([HB[wd][lB + 1] for wd in by_class[s]], 0) for s in SUPS], 0).astype(np.float64)
gm_sw = mus_sw.mean(0, keepdims=True)
_, _, Vt_sw = np.linalg.svd(mus_sw - gm_sw, full_matrices=False)
for rank in RANKS:
    if rank > HID: continue
    acc_r = []; acc_d = []
    U2 = Vt_sw[:min(rank, len(SUPS) - 1)].astype(np.float32)
    if U2.shape[0] == 0: continue
    for (rw, rs, dw, ds, sw, ss) in PAIRS:
        B_ = BASE[rw]
        h_rec = HB[rw][lB + 1].astype(np.float32); h_don = HB[dw][lB + 1].astype(np.float32)
        new = h_rec + proj(h_don - h_rec, U2)
        v1 = fwd(B_['pr'], patch=(lB, torch.tensor(new, device='cuda')))
        sr1, _ = score_rank(v1, rs, B_['sid_r'])
        sd1, _ = score_rank(v1, ds, B_['sid_d'])
        acc_r.append(sr1 - B_['sr0']); acc_d.append(sd1 - B_['sd0'])
    w('  rank=%-3d dDonor=%+.3f  dRecip=%+.3f  (n=%d)' %
      (rank, np.mean(acc_d), np.mean(acc_r), len(acc_d)))
    sys.stdout.flush()

w('')
w('total %.1fs' % (time.time() - t0))
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines))
print('DONE ->', OUT)
