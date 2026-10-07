# -*- coding: utf-8 -*-
"""
N2-h1b: 类别子空间的跨模型复现 / 跨层复用 / 边界诊断
=====================================================
N2h1 (qwen3-4b) 结论：读位槽类别归属由 5 维类别子空间承载（B/A=0.997），
                       2555 维正交残差零贡献，同类 donor 零效应，随机 5 维零效应。

本探针回答 5 问：
  A 边界诊断：最后一层 patch 的 S_self != 0 是 harness 伪影吗？（比较 hook 抓到的层输出
              与 output_hidden_states[-1] 的 cos / 范数比）
  B 跨模型：同样分解在 qwen2.5-3b / glm4-9b（untied）上成立吗？
  C 泛化：leave-one-class-out —— 用 5 个类的方向去移植第 6 类的实例，还有效吗？
  D 跨层复用：U_l（类别子空间）与 U_0（嵌入输出层）的主角度 —— 类别轴是否在 L0 就存在并被复用？
  E 必要性与方向身份：只置零 U 分量（vs 随机分量）会怎样？rank-1 的 5 个方向各是什么？
"""
import os, sys, time
import numpy as np
import torch

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'qwen2.5-3b-instruct'
MDIR = os.path.join(ROOT, 'models', 'hf', MODEL)
OUT = os.path.join(ROOT, 'tests', 'gpt5_temp', 'n2h1b_report_%s.txt' % MODEL.replace('/', '_'))
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
TMPL = '%s是一种'
PATCH_L = [1, 3, 5, 6, 7, 9, 12, 15, 17, 20, 24, 27, 30, 33, 34]
PATCH_L = [l for l in PATCH_L if l < L]

w('=== N2-h1b 类别子空间：跨模型 / 跨层复用 / 边界诊断 ===')
w('time %s model=%s L=%d hid=%d classes=%d inst=%d tie=%s' %
  (time.strftime('%Y-%m-%d %H:%M:%S'), MODEL, L, HID, len(SUPS), len(INST),
   model.config.tie_word_embeddings))
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
        l, vec = patch
        def hook(mod, args, out):
            if isinstance(out, tuple):
                h = out[0].clone(); h[0, -1, :] = vec.to(h.dtype)
                return (h,) + tuple(out[1:])
            h = out.clone(); h[0, -1, :] = vec.to(h.dtype); return h
        hs.append(layers[l].register_forward_hook(hook))
    out = model(input_ids=ii)
    for h in hs: h.remove()
    return out.logits[0, -1].float().detach().cpu().numpy()

@torch.no_grad()
def hidden_all(text):
    ii = torch.tensor([ids_of(text)], device='cuda')
    out = model(input_ids=ii, output_hidden_states=True)
    return np.stack([h[0, -1].float().detach().cpu().numpy() for h in out.hidden_states], 0)

# ---------- 采集 ----------
HB = {wd: hidden_all(TMPL % wd) for wd, _ in INST}
by_class = {}
for wd, sup in INST:
    by_class.setdefault(sup, []).append(wd)

def make_U(l, use_classes=None):
    cls = use_classes or SUPS
    mus = np.stack([np.mean([HB[wd][l + 1] for wd in by_class[s]], 0) for s in cls], 0).astype(np.float64)
    gm = mus.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(mus - gm, full_matrices=False)
    k = min(len(cls) - 1, 5)
    return Vt[:k].astype(np.float32)

UB = {l: make_U(l) for l in PATCH_L}
def proj(vec, U): return (vec @ U.T) @ U

# ---------- A: 边界诊断 ----------
w('')
w('--- A: 最后一层边界诊断（hook 层输出 vs output_hidden_states[-1]）---')
diag = []
for wd in [INST[i][0] for i in range(0, min(12, len(INST)))]:
    ii = torch.tensor([ids_of(TMPL % wd)], device='cuda')
    box = {}
    h = layers[L - 1].register_forward_hook(lambda m, a, o: box.__setitem__('v', (o[0] if isinstance(o, tuple) else o)[0, -1].detach().float().cpu().numpy()))
    with torch.no_grad():
        out = model(input_ids=ii, output_hidden_states=True)
    h.remove()
    v_hook = box['v']; v_hs = out.hidden_states[-1][0, -1].float().cpu().numpy()
    cs = float(v_hook @ v_hs / (np.linalg.norm(v_hook) * np.linalg.norm(v_hs) + 1e-12))
    diag.append((cs, np.linalg.norm(v_hook), np.linalg.norm(v_hs)))
w('  n=%d  mean cos(hook_layer_out, hidden_states[-1]) = %.6f' % (len(diag), np.mean([d[0] for d in diag])))
w('  norm ratio (hook/hs) mean = %.4f' % np.mean([d[1] / d[2] for d in diag]))
w('  => %s' % ('同源，L35 异常来自别处' if np.mean([d[0] for d in diag]) > 0.999 else
                '非同源！hidden_states[-1] 与层输出不同（推测为已过 final norm），末层 patch 应剔除'))
sys.stdout.flush()

# ---------- B: 跨模型承诺曲线 ----------
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
    PAIRS.append((rw, rs, dw, ds, sw))

BASE = {}
for (rw, rs, dw, ds, sw) in PAIRS:
    pr = TMPL % rw
    v0 = fwd(pr)
    sr0, _ = score_rank(v0, rs, ids_of(rw)[0])
    sd0, _ = score_rank(v0, ds, ids_of(dw)[0])
    BASE[rw] = dict(pr=pr, sr0=sr0, sd0=sd0)

def run_mode(mode, pairs=None, U_override=None):
    pairs = pairs or PAIRS
    res = {}
    for l in PATCH_L:
        Ub = U_override[l] if U_override is not None else UB[l]
        Vb = None
        if mode == 'D_rand':
            rq, _ = np.linalg.qr(RNG.standard_normal((Ub.shape[0], HID)).astype(np.float32).T)
            Vb = rq.T.astype(np.float32)
        dr_l = []; dd_l = []
        for (rw, rs, dw, ds, sw) in pairs:
            B = BASE[rw]
            src = rw if mode == 'S_self' else (sw if mode == 'E_same' else dw)
            h_rec = HB[rw][l + 1].astype(np.float32); h_don = HB[src][l + 1].astype(np.float32)
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
            v1 = fwd(B['pr'], patch=(l, torch.tensor(new, device='cuda')))
            sr1, _ = score_rank(v1, rs, ids_of(rw)[0])
            sd1, _ = score_rank(v1, ds, ids_of(dw)[0])
            dr_l.append(sr1 - B['sr0']); dd_l.append(sd1 - B['sd0'])
        res[l] = (float(np.mean(dr_l)), float(np.mean(dd_l)))
    return res

MODES = ['S_self', 'A_full', 'B_cat', 'C_resid', 'D_rand', 'E_same']
w('')
w('--- B: 承诺曲线 dDonor ---')
w('%-5s | %s' % ('L', '  '.join('%9s' % m for m in MODES)))
CURVE = {}
for m in MODES:
    CURVE[m] = run_mode(m)
for l in PATCH_L:
    w('%-5d | %s' % (l, '  '.join('%+9.3f' % CURVE[m][l][1] for m in MODES)))
A, B_, D_, E_, C_ = CURVE['A_full'], CURVE['B_cat'], CURVE['D_rand'], CURVE['E_same'], CURVE['C_resid']
jb = [(i, B_[l][1] - B_[PATCH_L[i - 1]][1]) for i, l in enumerate(PATCH_L) if i > 0]
i_j, jv = max(jb, key=lambda x: x[1])
lj = PATCH_L[i_j]
w('')
w('  B/A 末层 = %.3f ; 最大跳变 @L%d (+%.3f，前一层 L%d %+.3f)' %
  (B_[PATCH_L[-1]][1] / max(abs(A[PATCH_L[-1]][1]), 1e-9), lj, jv, PATCH_L[i_j - 1], B_[PATCH_L[i_j - 1]][1]))
w('  C末层=%+.3f  D末层=%+.3f  E末层=%+.3f  S末层=%+.3f' %
  (C_[PATCH_L[-1]][1], D_[PATCH_L[-1]][1], E_[PATCH_L[-1]][1], CURVE['S_self'][PATCH_L[-1]][1]))
sys.stdout.flush()

# ---------- C: leave-one-class-out ----------
w('')
w('--- C: leave-one-class-out（用 5 类的方向移植第 6 类实例）---')
locl = lj if lj in PATCH_L else PATCH_L[-1]
w('  测试层 = 承诺层 L%d' % locl)
for held in SUPS:
    use = [s for s in SUPS if s != held]
    U_ho = {l: (make_U(l, use) if l == locl else UB[l]) for l in PATCH_L}
    pairs_ho = [p for p in PAIRS if p[1] == held]
    if not pairs_ho: continue
    r = run_mode('B_cat', pairs=pairs_ho, U_override=U_ho)
    rA = run_mode('A_full', pairs=pairs_ho)
    rD = run_mode('D_rand', pairs=pairs_ho, U_override=U_ho)
    w('  hold-out %-6s n=%2d : B_cat=%+.3f  A_full=%+.3f  D_rand=%+.3f  B/A=%.2f' %
      (held, len(pairs_ho), r[locl][1], rA[locl][1], rD[locl][1],
       r[locl][1] / max(abs(rA[locl][1]), 1e-9)))
    sys.stdout.flush()

# ---------- D: 跨层子空间对齐 ----------
w('')
w('--- D: 类别子空间跨层复用（U_l vs U_0 主角度 cos 均值；主角度=1 表示同子空间）---')
U0 = make_U(0)
w('  U_0 rank=%d' % U0.shape[0])
w('%-5s %8s   %s' % ('L', 'mean|cos|', '5 个主角度'))
for l in PATCH_L:
    M = UB[l] @ U0.T
    sv = np.linalg.svd(M, compute_uv=False)
    w('%-5d %8.3f   %s' % (l, float(np.mean(sv)), ' '.join('%.3f' % x for x in sv)))
# 承诺层与全层对齐曲线（含 L0..L-1 快速扫描）
w('')
w('  --- 逐层 vs U_L%d（承诺层）---' % locl)
Ul = UB[locl]
sweep = list(range(0, L))
for l in sweep:
    U_ = make_U(l)
    sv = np.linalg.svd(U_ @ Ul.T, compute_uv=False)
    if l % 3 == 0 or l in (locl,):
        w('    L%-3d mean|cos|=%.3f  max=%.3f' % (l, float(np.mean(sv)), float(np.max(sv))))
sys.stdout.flush()

# ---------- E: 必要性（置零 U 分量）+ rank-1 方向身份 ----------
w('')
w('--- E1: 必要性 —— 只置零读位槽的 U 分量 vs 随机分量（不动其他 2555 维）---')
for l in [locl] + [x for x in [1, 5, 15, 30] if x in PATCH_L and x != locl]:
    acc_u = []; acc_r = []
    for (rw, rs, dw, ds, sw) in PAIRS:
        B = BASE[rw]
        h = HB[rw][l + 1].astype(np.float32)
        cm = proj(h, UB[l])
        v_u = fwd(B['pr'], patch=(l, torch.tensor(h - cm, device='cuda')))
        su, _ = score_rank(v_u, rs, ids_of(rw)[0])
        rq, _ = np.linalg.qr(RNG.standard_normal((UB[l].shape[0], HID)).astype(np.float32).T)
        Vb = rq.T.astype(np.float32)
        v_r = fwd(B['pr'], patch=(l, torch.tensor(h - proj(h, Vb), device='cuda')))
        sr, _ = score_rank(v_r, rs, ids_of(rw)[0])
        acc_u.append(su - B['sr0']); acc_r.append(sr - B['sr0'])
    w('  L%-3d 零化 U 分量 dScore=%+.3f | 零化随机分量 dScore=%+.3f' %
      (l, np.mean(acc_u), np.mean(acc_r)))
    sys.stdout.flush()

w('')
w('--- E2: rank-1 方向身份 @L%d（各主方向单独移植的效应 + 它偏向哪个类）---' % locl)
for i in range(min(5, UB[locl].shape[0])):
    u = UB[locl][i:i + 1]
    acc = []
    for (rw, rs, dw, ds, sw) in PAIRS:
        B = BASE[rw]
        h_rec = HB[rw][locl + 1].astype(np.float32); h_don = HB[dw][locl + 1].astype(np.float32)
        new = h_rec + proj(h_don - h_rec, u)
        v1 = fwd(B['pr'], patch=(locl, torch.tensor(new, device='cuda')))
        sd1, _ = score_rank(v1, ds, ids_of(dw)[0])
        acc.append(sd1 - B['sd0'])
    # 该方向对 6 类均值的投影
    mus = np.stack([np.mean([HB[wd][locl + 1] for wd in by_class[s]], 0) for s in SUPS], 0).astype(np.float32)
    pr_ = mus @ UB[locl][i]
    order = np.argsort(-pr_)
    w('  dir%d dDonor=%+.3f | 类投影 top: %s' %
      (i, np.mean(acc), ' > '.join('%s(%.1f)' % (SUPS[k], pr_[k]) for k in order[:3])))
    sys.stdout.flush()

w('')
w('total %.1fs' % (time.time() - t0))
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines))
print('DONE ->', OUT)
