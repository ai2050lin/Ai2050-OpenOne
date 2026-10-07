# -*- coding: utf-8 -*-
"""
N2-h1c: 类别子空间的"反循环论证"对照
=====================================
N2h1/h1b 的 U 是在 %s是一种 上下文下用【同一批实例】的类均值构造，又在同一上下文测试 —— 存在
in-sample 循环风险（LCO 只部分缓解）。本探针做三项独立化：

  P1 跨模板构造：U 用【这是{W}】构造，在【{W}是一种】上测（反之亦然）
  P2 留一实例：构造 U 时把被测实例从该类里移除（leave-instance-out）
  P3 跨任务：在另一 cloze 任务（{W}属于 / {W}是某种 / {W}和{co}都是）上测 B 模式效应
  P4 类均值方向 vs 随机方向对照（子空间内的方向是否都有效）

判据：若 P1/P2 下 B_cat 仍 >= 0.7 * A_full 且随机方向 ≈ 0 => 类别子空间是任务无关的稳定结构；
      若崩塌 => N2h1 的效应部分是"用同一批数据拟合的投影"，须降级。
"""
import os, sys, time
import numpy as np
import torch

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'qwen3-4b'
MDIR = os.path.join(ROOT, 'models', 'hf', MODEL)
OUT = os.path.join(ROOT, 'tests', 'gpt5_temp', 'n2h1c_report_%s.txt' % MODEL.replace('/', '_'))
lines = []
def w(s=''):
    lines.append(str(s)); print(s); sys.stdout.flush()

from transformers import AutoTokenizer, AutoModelForCausalLM
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MDIR, dtype=torch.bfloat16,
                                            trust_remote_code=True).to('cuda').eval()
_core = getattr(model.model, 'language_model', model.model)
layers = _core.layers; L = len(layers)
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
by_class = {}
for wd, sup in INST:
    by_class.setdefault(sup, []).append(wd)

CTX = {
    'cloze_N': ('%s是一种', 'N'),      # 用于测试
    'neutral': ('这是%s', 'B'),        # 用于构造 U
}
PROBE_TMPLS = {
    'cloze_N':  '%s是一种',
    'neutral':  '这是%s',
    'cloze_belong': '%s属于',
    'cloze_kind':   '%s是某种',
    'cloze_co':     '%s和桌子都是',
}
BUILD_TMPLS = {'cloze_N': '%s是一种', 'neutral': '这是%s'}

w('=== N2-h1c 反循环论证对照 ===')
w('time %s model=%s L=%d hid=%d inst=%d' % (time.strftime('%Y-%m-%d %H:%M:%S'), MODEL, L, HID, len(INST)))
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
def hidden_last(text):
    ii = torch.tensor([ids_of(text)], device='cuda')
    out = model(input_ids=ii, output_hidden_states=True)
    return np.stack([h[0, -1].float().detach().cpu().numpy() for h in out.hidden_states], 0)

# 每个实例在两种上下文下的全层末位状态（构造用）
HBUILD = {}
for tag, tmpl in BUILD_TMPLS.items():
    HBUILD[tag] = {wd: hidden_last(tmpl % wd) for wd, _ in INST}
w('states collected: %d instances x %d contexts' % (len(INST), len(HBUILD)))
sys.stdout.flush()

def make_U(l, ctx='cloze_N', drop_inst=None, classes=None):
    cls = classes or SUPS
    mus = []
    for s in cls:
        ms = [wd for wd in by_class[s] if wd != drop_inst]
        if not ms: return None
        mus.append(np.mean([HBUILD[ctx][wd][l + 1] for wd in ms], 0))
    mus = np.stack(mus, 0).astype(np.float64)
    gm = mus.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(mus - gm, full_matrices=False)
    k = min(len(cls) - 1, 5)
    return Vt[:k].astype(np.float32)

def proj(vec, U): return (vec @ U.T) @ U

# pair 构造（同 N2h1b）
n = len(INST)
PAIRS = []
for i, (rw, rs) in enumerate(INST):
    j = (i + n // 2) % n
    while INST[j][1] == rs:
        j = (j + 1) % n
    dw, ds = INST[j]
    PAIRS.append((rw, rs, dw, ds))

# 需要找承诺层：先用 cloze_N 构造+测试扫一遍
CANDS = [1, 2, 3, 4, 5, 6, 7, 9, 12, 15, 20, 25, 30, 34]
CANDS = [l for l in CANDS if l < L - 1]
w('')
w('--- 定位承诺层（A_full 末位 patch 的 dDonor 跳变）---')
W_L = '%s是一种'
BASE = {}
for (rw, rs, dw, ds) in PAIRS:
    v0 = fwd(W_L % rw)
    BASE[rw] = (score_rank(v0, rs, ids_of(rw)[0])[0], score_rank(v0, ds, ids_of(dw)[0])[0])
jump = {}
for l in CANDS:
    dd = []
    for (rw, rs, dw, ds) in PAIRS:
        h_don = HBUILD['cloze_N'][dw][l + 1].astype(np.float32)
        v1 = fwd(W_L % rw, patch=(l, torch.tensor(h_don, device='cuda')))
        dd.append(score_rank(v1, ds, ids_of(dw)[0])[0] - BASE[rw][1])
    jump[l] = float(np.mean(dd))
    w('  L%-3d A_full dDonor=%+.3f' % (l, jump[l]))
    sys.stdout.flush()
# 承诺层 = 相邻层最大增量处（不是最大值处 —— 之后是饱和平台）
inc = [(CANDS[i], jump[CANDS[i]] - jump[CANDS[i - 1]]) for i in range(1, len(CANDS))]
LSTAR, LSTAR_INC = max(inc, key=lambda x: x[1])
w('  承诺层 L*=%d（相邻最大增量 %+.3f；前一层 L%d = %+.3f）' %
  (LSTAR, LSTAR_INC, CANDS[CANDS.index(LSTAR) - 1], jump[CANDS[CANDS.index(LSTAR) - 1]]))
sys.stdout.flush()

# ---------- P1: 跨模板构造 ----------
w('')
w('--- P1: 跨模板构造 / 跨任务测试（承诺层 L%d）---' % LSTAR)
w('%-14s %-14s %10s %10s %10s %10s' % ('build_ctx', 'test_tmpl', 'B_cat', 'A_full', 'D_rand', 'B/A'))
SRC = 'cloze_N'
for build_tag in ['cloze_N', 'neutral']:
    U = make_U(LSTAR, ctx=build_tag)
    Ub = {LSTAR: U}
    for ptag, ptmpl in PROBE_TMPLS.items():
        # donor 状态仍取自 cloze_N（保持供体一致）
        accB = []; accA = []; accD = []
        rq, _ = np.linalg.qr(RNG.standard_normal((U.shape[0], HID)).astype(np.float32).T)
        Vb = rq.T.astype(np.float32)
        for (rw, rs, dw, ds) in PAIRS:
            pr = ptmpl % rw
            v0 = fwd(pr)
            sr0 = score_rank(v0, rs, ids_of(rw)[0])[0]
            sd0 = score_rank(v0, ds, ids_of(dw)[0])[0]
            h_rec = HBUILD['cloze_N'][rw][LSTAR + 1].astype(np.float32)
            h_don = HBUILD['cloze_N'][dw][LSTAR + 1].astype(np.float32)
            diff = h_don - h_rec
            for tag, new in [('B', h_rec + proj(diff, U)), ('A', h_don), ('D', h_rec + proj(diff, Vb))]:
                v1 = fwd(pr, patch=(LSTAR, torch.tensor(new, device='cuda')))
                sd1 = score_rank(v1, ds, ids_of(dw)[0])[0]
                (accB if tag == 'B' else accA if tag == 'A' else accD).append(sd1 - sd0)
        b, a, d = np.mean(accB), np.mean(accA), np.mean(accD)
        w('%-14s %-14s %10.3f %10.3f %10.3f %10.3f' % (build_tag, ptag, b, a, d, b / max(abs(a), 1e-9)))
        sys.stdout.flush()

# ---------- P2: 留一实例 ----------
w('')
w('--- P2: leave-instance-out（构造 U 时移除被测实例）---')
accB = []; accA = []
for (rw, rs, dw, ds) in PAIRS:
    U = make_U(LSTAR, ctx='cloze_N', drop_inst=rw)
    if U is None: continue
    pr = W_L % rw
    v0 = fwd(pr)
    sr0 = score_rank(v0, rs, ids_of(rw)[0])[0]; sd0 = score_rank(v0, ds, ids_of(dw)[0])[0]
    h_rec = HBUILD['cloze_N'][rw][LSTAR + 1].astype(np.float32)
    h_don = HBUILD['cloze_N'][dw][LSTAR + 1].astype(np.float32)
    v1 = fwd(pr, patch=(LSTAR, torch.tensor(h_rec + proj(h_don - h_rec, U), device='cuda')))
    accB.append(score_rank(v1, ds, ids_of(dw)[0])[0] - sd0)
    v2 = fwd(pr, patch=(LSTAR, torch.tensor(h_don, device='cuda')))
    accA.append(score_rank(v2, ds, ids_of(dw)[0])[0] - sd0)
w('  leave-inst-out: B_cat=%+.3f  A_full=%+.3f  B/A=%.3f' %
  (np.mean(accB), np.mean(accA), np.mean(accB) / max(abs(np.mean(accA)), 1e-9)))
sys.stdout.flush()

# ---------- P3: 跨层的 U（用 L0/L1 构造，承诺层测试）----------
w('')
w('--- P3: 用其他层的 U（含 L0/L1）在承诺层 L%d 测试 ---' % LSTAR)
for lsrc in [0, 1, 2, 3, 5, 10]:
    if lsrc >= L - 1: continue
    U = make_U(lsrc, ctx='cloze_N')
    accB = []; accA = []
    for (rw, rs, dw, ds) in PAIRS:
        pr = W_L % rw
        v0 = fwd(pr)
        sr0 = score_rank(v0, rs, ids_of(rw)[0])[0]; sd0 = score_rank(v0, ds, ids_of(dw)[0])[0]
        h_rec = HBUILD['cloze_N'][rw][LSTAR + 1].astype(np.float32)
        h_don = HBUILD['cloze_N'][dw][LSTAR + 1].astype(np.float32)
        diff = h_don - h_rec
        v1 = fwd(pr, patch=(LSTAR, torch.tensor(h_rec + proj(diff, U), device='cuda')))
        accB.append(score_rank(v1, ds, ids_of(dw)[0])[0] - sd0)
        v2 = fwd(pr, patch=(LSTAR, torch.tensor(h_don, device='cuda')))
        accA.append(score_rank(v2, ds, ids_of(dw)[0])[0] - sd0)
    w('  U from L%-3d : B_cat=%+.3f  A_full=%+.3f  B/A=%.3f' %
      (lsrc, np.mean(accB), np.mean(accA), np.mean(accB) / max(abs(np.mean(accA)), 1e-9)))
    sys.stdout.flush()

# ---------- P4: 子空间内随机方向 vs 类均值方向 ----------
w('')
w('--- P4: U 内随机方向 vs 类均值方向（5 维内是否每个方向都有效）---')
U = make_U(LSTAR, ctx='cloze_N')
for trial in range(4):
    r = RNG.standard_normal((1, U.shape[0])).astype(np.float32)
    r /= np.linalg.norm(r)
    u_mix = (r @ U).reshape(-1)           # U 内的随机方向（单位化）
    u_mix = (u_mix / np.linalg.norm(u_mix)).astype(np.float32).reshape(1, -1)
    acc = []
    for (rw, rs, dw, ds) in PAIRS:
        pr = W_L % rw
        v0 = fwd(pr); sd0 = score_rank(v0, ds, ids_of(dw)[0])[0]
        h_rec = HBUILD['cloze_N'][rw][LSTAR + 1].astype(np.float32)
        h_don = HBUILD['cloze_N'][dw][LSTAR + 1].astype(np.float32)
        v1 = fwd(pr, patch=(LSTAR, torch.tensor(h_rec + proj(h_don - h_rec, u_mix), device='cuda')))
        acc.append(score_rank(v1, ds, ids_of(dw)[0])[0] - sd0)
    w('  U-internal random dir #%d dDonor=%+.3f' % (trial, np.mean(acc)))
    sys.stdout.flush()
acc = []
for (rw, rs, dw, ds) in PAIRS:
    pr = W_L % rw
    v0 = fwd(pr); sd0 = score_rank(v0, ds, ids_of(dw)[0])[0]
    h_rec = HBUILD['cloze_N'][rw][LSTAR + 1].astype(np.float32)
    h_don = HBUILD['cloze_N'][dw][LSTAR + 1].astype(np.float32)
    v1 = fwd(pr, patch=(LSTAR, torch.tensor(h_rec + proj(h_don - h_rec, U), device='cuda')))
    acc.append(score_rank(v1, ds, ids_of(dw)[0])[0] - sd0)
w('  全 5 维 U（参照）    dDonor=%+.3f' % np.mean(acc))

w('')
w('total %.1fs' % (time.time() - t0))
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines))
print('DONE ->', OUT)
