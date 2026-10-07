# -*- coding: utf-8 -*-
"""
N3: 条件子空间装置的跨结构类型 / 跨极性 / 跨语言族复现
=====================================================
动机：外部方案的核心前提是"统一实验卡 + 每语言族测特征 + 拼图还原"。
      本项目唯一通过全部判据的装置是 N2h1 的「条件化类别子空间 + 置换向量消融」。
      本探针检验该装置能否 (a) 跨 L2 结构类型（is-a → 颜色属性）复现、
      (b) 跨极性（肯定 → 否定）保持、(c) 跨语言族迁移。

预注册：N3_design_seal.json（任何观测前冻结）
  F1 is-a  : {W}是一种        → 上位类 (6 类, 41 实例, 与 N2h1 同材料)
  F2 color : {W}的颜色是      → 颜色类 (6 类)
  F3 极性  : {W}不是一种      → 同一批实例，否定模板
  F4 跨族  : 用他族子空间对本族对做 B_cat + 子空间主角

模式：S_self / A_full / B_cat / B_cat_loo / C_resid / D_rand / E_same
"""
import os, sys, time, json
import numpy as np
import torch

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'qwen3-4b'
MDIR = os.path.join(ROOT, 'models', 'hf', MODEL)
OUT = os.path.join(ROOT, 'tests', 'gpt5_temp', 'n3_report_%s.txt' % MODEL.replace('/', '_'))
lines = []
def w(s=''):
    lines.append(str(s)); print(s); sys.stdout.flush()

from transformers import AutoTokenizer, AutoModelForCausalLM
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MDIR, dtype=torch.bfloat16,
                                            trust_remote_code=True).to('cuda').eval()
_core = getattr(model.model, 'language_model', model.model)
layers = _core.layers; L = len(layers); head = model.lm_head
HID = model.config.hidden_size
def ids_of(s): return tok.encode(s, add_special_tokens=False)

RNG = np.random.default_rng(20261001)

GROUPS_ISA = {
    '水果': ['苹果', '香蕉', '梨', '西瓜', '葡萄', '草莓', '芒果', '柠檬'],
    '动物': ['狗', '猫', '老虎', '大象', '兔子', '猴子', '马', '牛'],
    '交通工具': ['汽车', '火车', '飞机', '摩托车', '卡车', '地铁'],
    '家具': ['桌子', '椅子', '床', '沙发', '地毯', '窗帘'],
    '金属': ['铁', '铜', '铝', '金', '银', '锌', '铅'],
    '颜色': ['红', '蓝', '绿', '黄', '黑', '白'],
}
# F2 材料与 F1 完全无交集（避免跨族污染）
GROUPS_COL = {
    '红': ['番茄', '玫瑰', '辣椒', '消防车', '枫叶'],
    '蓝': ['天空', '大海', '牛仔裤', '蓝莓', '蓝宝石'],
    '绿': ['树叶', '青草', '翡翠', '黄瓜', '青蛙'],
    '黄': ['向日葵', '玉米', '蛋黄', '出租车'],
    '黑': ['煤炭', '墨水', '轮胎', '乌鸦', '黑板'],
    '白': ['雪', '云', '牛奶', '白纸', '月光'],
}
T_POS = '%s是一种'
T_NEG = '%s不是一种'
T_COL = '%s的颜色是'

SUP_ID = {s: ids_of(s)[0] for s in GROUPS_ISA if len(ids_of(s)) == 1}
SUPS = list(SUP_ID.keys())
INST_ISA = [(wd, sup) for sup, ms in GROUPS_ISA.items() if sup in SUP_ID
            for wd in ms if len(ids_of(wd)) == 1]
COL_ID = {c: ids_of(c)[0] for c in GROUPS_COL if len(ids_of(c)) == 1}
COLS = list(COL_ID.keys())
INST_COL = [(wd, c) for c, ms in GROUPS_COL.items() if c in COL_ID for wd in ms]

PATCH_L = [1, 3, 5, 6, 7, 9, 12, 15, 17, 20, 24, 27, 30, 33, 34]
PATCH_L = [l for l in PATCH_L if l < L - 1]
RANKS = [1, 2, 3, 5, 10, 20]

w('=== N3 条件子空间装置：跨结构 / 跨极性 / 跨族复现 ===')
w('seal N3_design_seal.json ; time %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
w('model=%s L=%d hid=%d tie=%s' % (MODEL, L, HID, model.config.tie_word_embeddings))
w('F1 is-a  classes=%d instances=%d' % (len(SUPS), len(INST_ISA)))
w('F2 color classes=%d instances=%d' % (len(COLS), len(INST_COL)))
w('patch layers=%s' % PATCH_L)
sys.stdout.flush()

def score(v, cmap, name, excl):
    v = v.copy()
    for i in excl:
        if 0 <= i < len(v):
            v[i] = -1e9
    own = v[cmap[name]]
    others = [v[cmap[x]] for x in cmap if x != name]
    order = np.argsort(-v)
    return float(own - np.mean(others)), int(np.where(order == cmap[name])[0][0]) + 1

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
    for h in hs:
        h.remove()
    return out.logits[0, -1].float().detach().cpu().numpy()

@torch.no_grad()
def hidden_all(text):
    ii = torch.tensor([ids_of(text)], device='cuda')
    out = model(input_ids=ii, output_hidden_states=True)
    return np.stack([h[0, -1].float().detach().cpu().numpy() for h in out.hidden_states], 0)

def proj(vec, U):
    return (vec @ U.T) @ U

def build_U(HB, items, cmap, l, excl=None, rank=None):
    mus = []
    for s in cmap:
        mem = [wd for (wd, c) in items if c == s and wd != excl]
        if not mem:
            return None, None
        mus.append(np.mean([HB[wd][l + 1] for wd in mem], 0))
    mus = np.stack(mus, 0).astype(np.float64)
    gm = mus.mean(0, keepdims=True)
    _, S, Vt = np.linalg.svd(mus - gm, full_matrices=False)
    k = rank if rank else min(len(mus) - 1, 5)
    return Vt[:k].astype(np.float32), S

def make_pairs(items):
    n = len(items); P = []
    for i, (rw, rc) in enumerate(items):
        j = (i + n // 2) % n
        while items[j][1] == rc:
            j = (j + 1) % n
        dw, dc = items[j]
        same = [x for x, c in items if c == rc and x != rw]
        sw = same[(i * 3) % len(same)] if same else None
        P.append((rw, rc, dw, dc, sw, rc))
    return P

# ---------------- 0. 采集 & 行为基线 ----------------
HISA, HNEG, HCOL = {}, {}, {}
for wd, _ in INST_ISA:
    HISA[wd] = hidden_all(T_POS % wd)
    HNEG[wd] = hidden_all(T_NEG % wd)
for wd, _ in INST_COL:
    HCOL[wd] = hidden_all(T_COL % wd)
w('hidden collected: isa %d (+neg) color %d' % (len(HISA), len(HCOL)))

w('')
w('--- 行为基线（贪心 top-1，sanity：模型是否真在做该任务）---')
for tag, sample in [('F1 %s是一种' % INST_ISA[0][0], T_POS % INST_ISA[0][0]),
                    ('F1 %s是一种' % INST_ISA[20][0], T_POS % INST_ISA[20][0]),
                    ('F3 %s不是一种' % INST_ISA[0][0], T_NEG % INST_ISA[0][0]),
                    ('F3 %s不是一种' % INST_ISA[20][0], T_NEG % INST_ISA[20][0]),
                    ('F2 %s的颜色是' % INST_COL[0][0], T_COL % INST_COL[0][0]),
                    ('F2 %s的颜色是' % INST_COL[15][0], T_COL % INST_COL[15][0])]:
    v = fwd(sample)
    top = np.argsort(-v)[:5]
    w('  %-18s -> %s' % (sample, ' | '.join('%s(%.1f)' % (tok.decode([int(i)]).replace(chr(10), ''), v[i]) for i in top)))
sys.stdout.flush()

# ---------------- 1. 子空间基准表 ----------------
UB = {'isa': {}, 'col': {}}
VB = {'isa': {}, 'col': {}}
for l in PATCH_L:
    for fam, HB, items, cmap in [('isa', HISA, INST_ISA, SUP_ID), ('col', HCOL, INST_COL, COL_ID)]:
        U, S = build_U(HB, items, cmap, l)
        UB[fam][l] = U; VB[fam][l] = None
        k = U.shape[0]
        rq, _ = np.linalg.qr(RNG.standard_normal((HID, k)).astype(np.float32))
        VB[fam][l] = rq.T.astype(np.float32)
w('')
w('subspace rank: isa=%d col=%d' % (UB['isa'][PATCH_L[0]].shape[0], UB['col'][PATCH_L[0]].shape[0]))
sys.stdout.flush()

# ---------------- 2. 三族 BASE ----------------
def make_base(pairs, HB, cmap, tmpl):
    B = {}
    for rw, rs, dw, ds, sw, ss in pairs:
        pr = tmpl % rw
        v0 = fwd(pr)
        cls_ids = set(cmap.values())
        er = set(i for i in ids_of(rw) if i not in cls_ids)
        ed = set(i for i in ids_of(dw) if i not in cls_ids)
        sr0, rr0 = score(v0, cmap, rs, er)
        sd0, rd0 = score(v0, cmap, ds, ed)
        B[rw] = dict(pr=pr, er=er, ed=ed, sr0=sr0, sd0=sd0, rr0=rr0, rd0=rd0)
    return B

PAIRS_ISA = make_pairs(INST_ISA)
PAIRS_COL = make_pairs(INST_COL)
BASE_POS = make_base(PAIRS_ISA, HISA, SUP_ID, T_POS)
BASE_NEG = make_base(PAIRS_ISA, HNEG, SUP_ID, T_NEG)
BASE_COL = make_base(PAIRS_COL, HCOL, COL_ID, T_COL)
w('')
w('base recipient-class score: F1pos=%.3f  F3neg=%.3f  F2col=%.3f' % (
   np.mean([BASE_POS[r]['sr0'] for r, *_ in PAIRS_ISA]),
   np.mean([BASE_NEG[r]['sr0'] for r, *_ in PAIRS_ISA]),
   np.mean([BASE_COL[r]['sr0'] for r, *_ in PAIRS_COL])))
sys.stdout.flush()

MODES = ['S_self', 'A_full', 'B_cat', 'C_resid', 'D_rand', 'E_same']

def run_layer(l, HB, cmap, BASE, pairs, U, V, modes, donor_from=None):
    HBd = HB if donor_from is None else donor_from
    res = {}
    for m in modes:
        dr = []; dd = []
        for rw, rs, dw, ds, sw, ss in pairs:
            B = BASE[rw]
            src = rw if m == 'S_self' else (sw if m == 'E_same' else dw)
            h_rec = HB[rw][l + 1].astype(np.float32)
            h_don = HBd[src][l + 1].astype(np.float32)
            diff = h_don - h_rec
            if m == 'A_full':
                new = h_don
            elif m == 'S_self':
                new = h_don
            elif m == 'B_cat':
                new = h_rec + proj(diff, U)
            elif m == 'C_resid':
                new = h_rec + (diff - proj(diff, U))
            elif m == 'D_rand':
                new = h_rec + proj(diff, V)
            elif m == 'E_same':
                new = h_rec + proj(diff, U)
            elif m == 'B_loo':
                Ul, _ = build_U(HB, [(a, b) for a, b in zip([p[0] for p in pairs], [p[1] for p in pairs])], cmap, l, excl=rw)
                new = h_rec + proj(diff, Ul)
            elif m == 'B_wrong':
                new = h_rec + proj(diff, U)
            v1 = fwd(B['pr'], patch=(l, torch.tensor(new, device='cuda')))
            sr1, _ = score(v1, cmap, rs, B['er'])
            sd1, _ = score(v1, cmap, ds, B['ed'])
            dr.append(sr1 - B['sr0']); dd.append(sd1 - B['sd0'])
        res[m] = (float(np.mean(dr)), float(np.mean(dd)), len(dr))
    return res

# ---------------- 3. F1 / F2 全层扫描 ----------------
CURVE = {}
for fam, HB, cmap, BASE, pairs, itm in [('isa', HISA, SUP_ID, BASE_POS, PAIRS_ISA, INST_ISA),
                                        ('col', HCOL, COL_ID, BASE_COL, PAIRS_COL, INST_COL)]:
    CURVE[fam] = {}
    for m in ['S_self', 'A_full', 'B_cat', 'D_rand', 'E_same']:
        cv = {}
        for l in PATCH_L:
            cv[l] = run_layer(l, HB, cmap, BASE, pairs, UB[fam][l], VB[fam][l], [m])[m]
        CURVE[fam][m] = cv
        w('  [%s %s] done @%.0fs' % (fam, m, time.time() - t0)); sys.stdout.flush()

for fam, itm in [('isa', INST_ISA), ('col', INST_COL)]:
    w('')
    w('--- %s 承诺曲线（dDonor）---' % fam)
    w('%-5s | %s' % ('L', '  '.join('%9s' % m for m in ['S_self', 'A_full', 'B_cat', 'D_rand', 'E_same'])))
    for l in PATCH_L:
        w('%-5d | %s' % (l, '  '.join('%+9.3f' % CURVE[fam][m][l][1] for m in ['S_self', 'A_full', 'B_cat', 'D_rand', 'E_same'])))

# ---------------- 4. 承诺层定位（相邻最大增量）----------------
LSTAR = {}
for fam in ['isa', 'col']:
    cv = CURVE[fam]['B_cat']
    jumps = [(l, cv[l][1] - cv[PATCH_L[i - 1]][1]) for i, l in enumerate(PATCH_L) if i > 0]
    lj, jv = max(jumps, key=lambda x: x[1])
    LSTAR[fam] = lj
    w('')
    w('%s 承诺层 lstar=L%d  (相邻增量 %+.3f ; 前一层 %+.3f)' % (fam, lj, jv, cv[PATCH_L[PATCH_L.index(lj) - 1]][1]))
    w('   A_full 相邻增量 @L%d = %+.3f' % (lj, CURVE[fam]['A_full'][lj][1] - CURVE[fam]['A_full'][PATCH_L[PATCH_L.index(lj) - 1]][1]))
sys.stdout.flush()

# ---------------- 5. 承诺层详细模式 ----------------
w('')
w('--- 承诺层详细模式 ---')
DET = {}
for fam, HB, cmap, BASE, pairs in [('isa', HISA, SUP_ID, BASE_POS, PAIRS_ISA),
                                   ('col', HCOL, COL_ID, BASE_COL, PAIRS_COL)]:
    l = LSTAR[fam]
    r = run_layer(l, HB, cmap, BASE, pairs, UB[fam][l], VB[fam][l], ['S_self', 'A_full', 'B_cat', 'C_resid', 'D_rand', 'E_same'])
    r['B_loo'] = run_layer(l, HB, cmap, BASE, pairs, UB[fam][l], VB[fam][l], ['B_loo'])[ 'B_loo']
    DET[fam] = r
    A = abs(r['A_full'][1])
    w('  %s @L%d : A_full=%+.3f  B_cat=%+.3f  B_loo=%+.3f  C_resid=%+.3f  D_rand=%+.3f  E_same=%+.3f  S_self=%+.3f' % (
        fam, l, r['A_full'][1], r['B_cat'][1], r['B_loo'][1], r['C_resid'][1], r['D_rand'][1], r['E_same'][1], r['S_self'][1]))
    w('     B/A=%.3f  B_loo/B=%.3f  E/B=%.3f  D/B=%.3f' % (
        r['B_cat'][1] / max(A, 1e-9), r['B_loo'][1] / max(abs(r['B_cat'][1]), 1e-9),
        r['E_same'][1] / max(abs(r['B_cat'][1]), 1e-9), r['D_rand'][1] / max(abs(r['B_cat'][1]), 1e-9)))
sys.stdout.flush()

# ---------------- 6. K_d: 极性解耦（否定模板）----------------
w('')
w('--- K_d 极性解耦：否定模板 {W}不是一种 @isa lstar=L%d ---' % LSTAR['isa'])
l = LSTAR['isa']
# 供体状态仍取肯定上下文（跨上下文移植）
r_neg = run_layer(l, HNEG, SUP_ID, BASE_NEG, PAIRS_ISA, UB['isa'][l], VB['isa'][l], ['A_full', 'B_cat', 'D_rand'])
# 供体取肯定上下文
r_negX = run_layer(l, HNEG, SUP_ID, BASE_NEG, PAIRS_ISA, UB['isa'][l], VB['isa'][l], ['B_cat'], donor_from=HISA)
DET['neg'] = r_neg; DET['negX'] = r_negX
w('  同上下文(neg donor->neg rec): A_full=%+.3f  B_cat=%+.3f  D_rand=%+.3f  B/A=%.3f' % (
   r_neg['A_full'][1], r_neg['B_cat'][1], r_neg['D_rand'][1],
   r_neg['B_cat'][1] / max(abs(r_neg['A_full'][1]), 1e-9)))
w('  跨上下文(pos donor->neg rec): B_cat=%+.3f' % r_negX['B_cat'][1])
sys.stdout.flush()

# ---------------- 7. F4 跨族 ----------------
w('')
w('--- F4 跨族：子空间主角 + 错族子空间移植 ---')
for l in PATCH_L:
    M = UB['isa'][l] @ UB['col'][l].T
    sv = np.linalg.svd(M, compute_uv=False)
    w('  L%-3d 主角 cos = %s' % (l, ' '.join('%.3f' % x for x in sv)))
A2 = UB['isa'][LSTAR['col']]
r_wrong = run_layer(LSTAR['col'], HCOL, COL_ID, BASE_COL, PAIRS_COL, A2, VB['col'][LSTAR['col']], ['B_wrong'])
DET['col_wrong'] = r_wrong
w('  col @L%d 用 isa 子空间: B_wrong=%+.3f  vs B_cat=%+.3f  ratio=%.3f' % (
   LSTAR['col'], r_wrong['B_wrong'][1], DET['col']['B_cat'][1],
   r_wrong['B_wrong'][1] / max(abs(DET['col']['B_cat'][1]), 1e-9)))
sys.stdout.flush()

# ---------------- 8. rank sweep ----------------
w('')
w('--- rank sweep @承诺层（B 模式）---')
for fam, HB, cmap, BASE, pairs, itm in [('isa', HISA, SUP_ID, BASE_POS, PAIRS_ISA, INST_ISA),
                                        ('col', HCOL, COL_ID, BASE_COL, PAIRS_COL, INST_COL)]:
    l = LSTAR[fam]
    w('  [%s] @L%d' % (fam, l))
    for rank in RANKS:
        U2, _ = build_U(HB, itm, cmap, l, rank=rank)
        if U2 is None:
            continue
        r = run_layer(l, HB, cmap, BASE, pairs, U2, VB[fam][l], ['B_cat'])
        w('    rank=%-3d (实际 k=%d) dDonor=%+.3f' % (rank, U2.shape[0], r['B_cat'][1]))
        sys.stdout.flush()

# ---------------- 9. 判据裁决 ----------------
w('')
w('--- 预注册判据裁决 ---')
def verdict():
    di = DET['isa']; dc = DET['col']
    A_i = abs(di['A_full'][1]); A_c = abs(dc['A_full'][1])
    rB_i = di['B_cat'][1] / max(A_i, 1e-9); rB_c = dc['B_cat'][1] / max(A_c, 1e-9)
    w('F1 is-a  B/A=%.3f (lstar=L%d)' % (rB_i, LSTAR['isa']))
    w('F2 color B/A=%.3f (lstar=L%d)' % (rB_c, LSTAR['col']))
    jc = CURVE['col']['B_cat'][LSTAR['col']][1] - CURVE['col']['B_cat'][PATCH_L[PATCH_L.index(LSTAR['col']) - 1]][1]
    ji = CURVE['isa']['B_cat'][LSTAR['isa']][1] - CURVE['isa']['B_cat'][PATCH_L[PATCH_L.index(LSTAR['isa']) - 1]][1]
    mx_i = max(abs(CURVE['isa']['B_cat'][l][1]) for l in PATCH_L)
    mx_c = max(abs(CURVE['col']['B_cat'][l][1]) for l in PATCH_L)
    w('[诊断·非判据] 归一化跳变 jump/max：isa=%.3f ; col=%.3f' % (ji / max(mx_i, 1e-9), jc / max(mx_c, 1e-9)))
    w('   => 绝对跳变阈值 >3.0 在 is-a 尺度上标定，跨族沿用即"统一指标"陷阱')
    w('K_a 装置跨 L2 结构复现: B/A=%.3f (>=0.70?) ; 跳变=%+.3f (>3.0?) ; D/B=%.3f (<1.0?) -> %s' % (
        rB_c, jc, dc['D_rand'][1] / max(abs(dc['B_cat'][1]), 1e-9),
        'PASS' if (rB_c >= 0.70 and jc > 3.0 and abs(dc['D_rand'][1]) < abs(dc['B_cat'][1])) else 'FAIL'))
    w('K_b 类别特异: isa E/B=%.3f ; col E/B=%.3f (<0.30?)' % (
        di['E_same'][1] / max(abs(di['B_cat'][1]), 1e-9), dc['E_same'][1] / max(abs(dc['B_cat'][1]), 1e-9)))
    w('K_c 留一泛化: isa B_loo/B=%.3f ; col B_loo/B=%.3f (>=0.70?)' % (
        di['B_loo'][1] / max(abs(di['B_cat'][1]), 1e-9), dc['B_loo'][1] / max(abs(dc['B_cat'][1]), 1e-9)))
    rn = DET['neg']
    w('K_d 极性解耦(否定): B/A=%.3f ; 跨上下文 B_cat=%+.3f 同号? %s' % (
        rn['B_cat'][1] / max(abs(rn['A_full'][1]), 1e-9), DET['negX']['B_cat'][1],
        'YES' if (rn['B_cat'][1] * rn['A_full'][1] > 0) else 'NO'))
    rw_ = DET['col_wrong']
    w('K_e 跨族: col 用 isa 子空间 ratio=%.3f (<0.30=族特异 ; >=0.70=共享)' % (
        rw_['B_wrong'][1] / max(abs(dc['B_cat'][1]), 1e-9)))
    w('')
    w('>>> 总结：装置跨结构类型=%s ; 跨极性=%s ; 跨族共享=%s' % (
        'PASS' if rB_c >= 0.70 else 'FAIL',
        'PASS' if (rn['B_cat'][1] * rn['A_full'][1] > 0 and rn['B_cat'][1] / max(abs(rn['A_full'][1]), 1e-9) >= 0.70) else 'FAIL',
        'SHARED' if rw_['B_wrong'][1] / max(abs(dc['B_cat'][1]), 1e-9) >= 0.70 else 'FAMILY-SPECIFIC'))
verdict()

w('')
w('total %.1fs' % (time.time() - t0))
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines))
print('DONE ->', OUT)
