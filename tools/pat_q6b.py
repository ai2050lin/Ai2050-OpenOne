# -*- coding: utf-8 -*-
"""Q6-B：属性知识的秩与解耦（「有限参数记无限属性」的可判决形式）。

预注册：research/gpt5/docs/PAT_Q6B_PREREG_v1.md
        §1–§5 原判据 + §6 跑前修订（§6.1 双中心化 / §6.2 留出格子 /
        §6.4 打分口径改单 token 条件概率 / §6.5 三条前置 / §6.6 判决补档 / §6.7 零分布）

Q1（预测型）  物体×属性值矩阵 M[o,a]=log p(a | 前缀) 能压缩到几维？按格子 5 折补全
Q2（剖面型）  属性值向量落在「物体前 3 主成分」之外的份额，是否高于同词性形容词？

五个会把结论做成自动成立的坑（§6 全部已修）
  ① 只扣列均值 ⇒ row_o 独占方差 ⇒ R*=1 ⇒ 3D 必然成立
  ② 留出整个物体 ⇒ row_o 不可知 ⇒ R² 恒负
  ③ 续写打分被分词合并（`很大` 是一个 token）⇒ 大小门 0/8，等于没测
  ④ 只报标签门 ⇒ 测的是我的标注，不是测量可靠性
  ⑤ 零分布从物体池抽词 ⇒ dec_null 系统性偏低 ⇒ 假性 DECOUPLED

用法
----
  SMOKE=1 .venv/Scripts/python.exe -X utf8 tools/pat_q6b.py
  .venv/Scripts/python.exe -X utf8 tools/pat_q6b.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / 'tools')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import provenance
from pat_q1_unbind import render
from pat_q6a import MODEL_PATH, SEED, word_span

SMOKE = os.environ.get('SMOKE', '') == '1'

# ---------------- 材料（预注册 §1，冻结） ----------------
OBJ = ['苹果', '香蕉', '葡萄', '橘子', '白菜', '土豆', '黄瓜', '萝卜',
       '狗', '猫', '马', '鸟', '鱼', '老虎', '蚂蚁', '蚊子', '鲸鱼',
       '刀', '笔', '杯子', '钉子', '雪', '煤', '天空', '草', '西瓜', '老鼠', '楼房']
COLORS = ['红', '黄', '绿', '白', '黑', '蓝', '紫', '橙']
SIZES = ['大', '小']
TEMPS = ['热', '冷']
ATTR = COLORS + SIZES + TEMPS
TYPES = {'color': COLORS, 'size': SIZES, 'temp': TEMPS}
A2T = {a: t for t, vs in TYPES.items() for a in vs}

# 前缀必须以属性值之前的位置结束（§6.4）
PREFIX = {
    'color': ['「{O}」通常是', '按照常识，{O}的颜色是'],
    'size': ['「{O}」通常很', '{O}的体型很'],
    'temp': ['「{O}」摸起来是', '{O}的温度是'],
}

COLOR_PROBE = {'苹果': '红', '香蕉': '黄', '草': '绿', '雪': '白',
               '煤': '黑', '天空': '蓝', '葡萄': '紫', '橘子': '橙'}
SIZE_PROBE = {'蚂蚁': '小', '老鼠': '小', '西瓜': '大', '鲸鱼': '大',
              '钉子': '小', '蚊子': '小', '老虎': '大', '楼房': '大'}

# 零分布池：与属性值同词性同字长，但不是被测的 12 个（§6.7）
ADJ_NULL = ['美', '丑', '快', '慢', '高', '矮', '新', '旧', '轻', '重', '软', '硬',
            '甜', '苦', '香', '臭', '远', '近', '亮', '暗', '干', '湿', '厚', '薄']

TH_LABEL_C, TH_LABEL_S = 5, 6
TH_TPL = 0.50
TH_CONTRAST = 1.0
TH_R2 = 0.70
TH_DELTA = 0.15
TH_DEC = 0.15
TH_FRAC = 0.30
K_OBJ_PC = 3
N_NULL_DEC = 10


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def nextlogp(model, tok, prefix_text, cand_ids, dev):
    """在 prefix 末位读出各候选 token 的 log-prob（单 token 口径，§6.4）。"""
    import torch
    full = render(tok, prefix_text)
    ids = tok(full, add_special_tokens=False)['input_ids']
    with torch.no_grad():
        lg = model(input_ids=torch.tensor([ids], device=dev), use_cache=False).logits[0].float()
    lp = torch.log_softmax(lg[-1], dim=-1)
    return np.array([float(lp[i].item()) for i in cand_ids])


def soft_impute(X, mask, R, iters=25):
    """main effects + rank-R 残差迭代补全；只用 mask=True 的格子。"""
    mu = X[mask].mean()
    Xf = np.where(mask, X, mu)
    for _ in range(iters):
        r = np.array([(Xf[i][mask[i]].mean() - mu) if mask[i].any() else 0.0
                      for i in range(X.shape[0])])
        c = np.array([(Xf[:, j][mask[:, j]].mean() - mu) if mask[:, j].any() else 0.0
                      for j in range(X.shape[1])])
        Z = Xf - mu - r[:, None] - c[None, :]
        if R <= 0:
            Zr = np.zeros_like(Z)
        else:
            U, S, Vt = np.linalg.svd(Z, full_matrices=False)
            Zr = (U[:, :R] * S[:R]) @ Vt[:R]
        Xf = np.where(mask, X, mu + r[:, None] + c[None, :] + Zr)
    r = np.array([(Xf[i][mask[i]].mean() - mu) if mask[i].any() else 0.0
                  for i in range(X.shape[0])])
    c = np.array([(Xf[:, j][mask[:, j]].mean() - mu) if mask[:, j].any() else 0.0
                  for j in range(X.shape[1])])
    Z = Xf - mu - r[:, None] - c[None, :]
    if R > 0:
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        Zr = (U[:, :R] * S[:R]) @ Vt[:R]
    else:
        Zr = np.zeros_like(Z)
    return mu + r[:, None] + c[None, :] + Zr


def cv_r2(X, n_fold=5, rng_=None, shuffle=False):
    n, m = X.shape
    Xuse = X.copy()
    if shuffle:
        flat = X.ravel().copy()
        rng_.shuffle(flat)
        Xuse = flat.reshape(n, m)
    order = rng_.permutation(n * m)
    folds = np.array_split(order, n_fold)
    Rs = list(range(0, 8)) + [9, 11]
    r2 = {}
    for R in Rs:
        out_o, out_p = [], []
        for f in folds:
            mask = np.ones((n, m), dtype=bool)
            mask.ravel()[f] = False
            mask &= ~np.isnan(Xuse)
            pred = soft_impute(Xuse, mask, R)
            out_o.append(Xuse.ravel()[f])
            out_p.append(pred.ravel()[f])
        o_ = np.concatenate(out_o)
        p_ = np.concatenate(out_p)
        sst = ((o_ - o_.mean()) ** 2).sum()
        r2[R] = float(1.0 - ((o_ - p_) ** 2).sum() / max(sst, 1e-12))
    return r2


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    rng = np.random.default_rng(SEED)
    dev = 'cuda'
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, attn_implementation='eager').to(dev).eval()

    tmpls = {k: v[:1] for k, v in PREFIX.items()} if SMOKE else PREFIX
    aid = {a: tok(a, add_special_tokens=False)['input_ids'][0] for a in ATTR}
    assert all(len(tok(a, add_special_tokens=False)['input_ids']) == 1 for a in ATTR), \
        '属性值必须全为单 token（§6.4）'

    # ---------- Q1：行为矩阵（单 token 条件概率口径） ----------
    Ms = {k: np.zeros((len(OBJ), len(TYPES[k]))) for k in TYPES}
    for oi, o in enumerate(OBJ):
        for ty, vals in TYPES.items():
            ids = [aid[a] for a in vals]
            rows = []
            for p in tmpls[ty]:
                rows.append(nextlogp(model, tok, p.replace('{O}', o), ids, dev))
            Ms[ty][oi] = np.mean(rows, axis=0)
        if oi % 9 == 0:
            print('  M %d/%d  %.0fs' % (oi + 1, len(OBJ), time.time() - t0))
    M = np.concatenate([Ms[k] for k in ('color', 'size', 'temp')], axis=1)
    print('矩阵完成 %.0fs  shape=%s' % (time.time() - t0, M.shape))

    # ---------- 前置条件 C1/C2/C3 ----------
    cg = sum(1 for o, c in COLOR_PROBE.items()
             if COLORS[int(np.argmax(Ms['color'][OBJ.index(o)]))] == c)
    sg = sum(1 for o, s in SIZE_PROBE.items()
             if SIZES[int(np.argmax(Ms['size'][OBJ.index(o)]))] == s)
    print('\nC1 标签门：颜色 %d/8（阈 %d）、大小 %d/8（阈 %d）'
          % (cg, TH_LABEL_C, sg, TH_LABEL_S))
    for o, c in COLOR_PROBE.items():
        r = Ms['color'][OBJ.index(o)]
        print('   %s → %s（真值 %s，margin %.2f）'
              % (o, COLORS[int(np.argmax(r))], c, float(np.sort(r)[-1] - np.sort(r)[-2])))

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        if ra.std() < 1e-9 or rb.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(ra, rb)[0, 1])

    if len(tmpls['color']) >= 2:
        c2 = np.array([spearman(
            nextlogp(model, tok, tmpls['color'][0].replace('{O}', o),
                     [aid[a] for a in COLORS], dev),
            nextlogp(model, tok, tmpls['color'][1].replace('{O}', o),
                     [aid[a] for a in COLORS], dev)) for o in OBJ])
        c2_med = float(np.median(c2))
    else:
        c2_med = float('nan')
    c3 = float(np.median([float(Ms[k][i].max() - Ms[k][i].min())
                          for k in TYPES for i in range(len(OBJ))]))
    print('C2 模板一致（颜色 8 维 Spearman）中位 %.2f（阈 %.2f）' % (c2_med, TH_TPL))
    print('C3 属性类型内对比度 中位 %.2f nat（阈 %.1f）' % (c3, TH_CONTRAST))
    reliable = (cg >= TH_LABEL_C and sg >= TH_LABEL_S
                and (SMOKE or (c2_med >= TH_TPL)) and c3 >= TH_CONTRAST)
    print('前置条件 %s' % ('全过' if reliable else '未过 ⇒ MEASUREMENT_UNRELIABLE，不判 Q1'))

    # ---------- Q1：秩曲线 ----------
    r2_true = cv_r2(M, rng_=rng)
    r2_null = cv_r2(M, rng_=np.random.default_rng(SEED + 1), shuffle=True)
    print('\n=== Q1 秩曲线（按格子 5 折；R=0 = 只有主效应）===')
    for R in sorted(r2_true):
        print('  R=%-2d  R²%+.3f   打乱零分布%+.3f' % (R, r2_true[R], r2_null[R]))
    Rs = sorted(r2_true)
    Rbest = max(Rs, key=lambda R: r2_true[R])
    Rstar = next((R for R in Rs if r2_true[R] >= TH_R2), None)
    print('R² 最大在 R=%d（%+.3f）；R*=%s' % (Rbest, r2_true[Rbest], Rstar))
    if not reliable:
        q1 = 'MEASUREMENT_UNRELIABLE'
    elif Rbest == 0 or r2_true[11] < TH_R2:
        q1 = 'NO_LOW_RANK_STRUCTURE'
    elif Rstar is not None and Rstar <= 3 and r2_true[3] - r2_true[0] >= TH_DELTA:
        q1 = 'ATTRIBUTE_SPACE_3D'
    elif Rstar is not None and 4 <= Rstar <= 7:
        q1 = 'ATTRIBUTE_SPACE_COMPACT'
    else:
        q1 = 'FULL_RANK'
    print('R²(3)−R²(0)=%+.3f ⇒ Q1 = %s' % (r2_true[3] - r2_true[0], q1))

    # ---------- 诊断（不给判据）：方差分解 + 尖峰度 ----------
    # 若 interaction 占方差大头而 rank 越高越差 ⇒ 结构是「稀疏」不是「低秩」，
    # 低秩补全对稀疏结构必然失败。这两条数字是解释 R² 曲线形状的唯一依据。
    mu = M.mean()
    rowm = M.mean(1) - mu
    colm = M.mean(0) - mu
    inter = M - mu - rowm[:, None] - colm[None, :]
    vt = ((M - mu) ** 2).sum()
    # 两因素分解：Σ(M−μ)² = m·Σr² + n·Σc² + Σz²（r,c 已中心化）⇒ 必须乘对方维度数
    v_row = M.shape[1] * (rowm ** 2).sum()
    v_col = M.shape[0] * (colm ** 2).sum()
    v_int = ((inter - inter.mean()) ** 2).sum()
    print('\n方差分解：row %.2f  col %.2f  interaction %.2f'
          % (v_row / vt, v_col / vt, v_int / vt))

    peak = []
    for k in TYPES:
        Xk = Ms[k]
        ps = (Xk.max(1) - np.median(Xk, 1)) / np.maximum(Xk.max(1) - Xk.min(1), 1e-9)
        peak.append(ps)
    peak = np.concatenate(peak)
    n_above = np.concatenate([(Ms[k] >= Ms[k].max(1, keepdims=True) - 1.0).sum(1)
                              for k in TYPES])
    print('尖峰度（(max−median)/(max−min)）中位 %.2f；'
          '每物体每属性类型「距峰值 1 nat 内」的值个数 中位 %.1f / 平均 %.2f'
          % (float(np.median(peak)), float(np.median(n_above)), float(np.mean(n_above))))
    print('⇒ 若尖峰度高而低秩失败，结构就是稀疏码（少量强属性）而非低秩码')

    # ---------- Q2：解耦 ----------
    vocab = list(dict.fromkeys(OBJ + ATTR + ADJ_NULL))
    V = {}
    base_t = ['我想到一个词：{W}。'] if SMOKE else ['我想到一个词：{W}。', '他刚刚提到了{W}。']
    NL = model.config.num_hidden_layers + 1
    with torch.no_grad():
        for w in vocab:
            acc = []
            for t in base_t:
                user = t.replace('{W}', w)
                full = render(tok, user)
                b = full.find(user)
                if b < 0:
                    continue
                c0 = t.index('{W}')
                sp = word_span(tok, full, b, c0, c0 + len(w))
                if not sp:
                    continue
                ids = tok(full, add_special_tokens=False)['input_ids']
                out = model(input_ids=torch.tensor([ids], device=dev),
                            use_cache=False, output_hidden_states=True)
                acc.append(np.stack([out.hidden_states[li][0, sp, :].float().mean(0).cpu().numpy()
                                     for li in range(NL)]))
            V[w] = np.mean(acc, axis=0) if acc else None
    ObjV = np.stack([V[o] for o in OBJ])
    AttrV = np.stack([V[a] for a in ATTR])
    AdjV = np.stack([V[a] for a in ADJ_NULL if V[a] is not None])

    def dec_of(Xa, P):
        Xc = Xa - Xa.mean(0)
        return 1.0 - ((Xc @ P) ** 2).sum() / max((Xc ** 2).sum(), 1e-12)

    dec_p, dec_null_p = np.zeros(NL), np.zeros(NL)
    for li in range(NL):
        Xo = ObjV[:, li, :]
        U, S, Vt = np.linalg.svd(Xo - Xo.mean(0), full_matrices=False)
        P = Vt[:K_OBJ_PC].T
        dec_p[li] = dec_of(AttrV[:, li, :], P)
        dec_null_p[li] = dec_of(AdjV[:, li, :], P)
    dif = dec_p - dec_null_p
    f_pos_d = float(np.mean(dif >= TH_DEC))
    f_neg_d = float(np.mean(dif <= -TH_DEC))
    q2 = ('DECOUPLED' if f_pos_d >= TH_FRAC else
          'COUPLED' if f_neg_d >= TH_FRAC else 'NOT_SEPARABLE')
    print('\n=== Q2 解耦（属性值 vs 物体前 %d 主成分；零分布=同词性形容词）=== 层数 %d'
          % (K_OBJ_PC, NL))
    print('dec       %s' % ' '.join('%.2f' % x for x in dec_p))
    print('dec_adj   %s' % ' '.join('%.2f' % x for x in dec_null_p))
    print('diff      %s' % ' '.join('%+.2f' % x for x in dif))
    print('diff≥+%.2f 层占比 %.2f，≤−%.2f %.2f ⇒ Q2 = %s'
          % (TH_DEC, f_pos_d, TH_DEC, f_neg_d, q2))

    # ---------- 交叉检查（不给判据） ----------
    X = M - M.mean(0, keepdims=True)
    X = X - X.mean(1, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    print('\n交叉检查：interaction 前 3 奇异值占比 %s'
          % ' '.join('%.2f' % (s ** 2 / (S ** 2).sum()) for s in S[:3]))
    for r in range(3):
        lo = Vt[r]
        print('  PC%d：颜色 %s | 大小 %s | 温度 %s'
              % (r + 1,
                 ' '.join('%+.2f' % lo[i] for i in range(len(COLORS))),
                 ' '.join('%+.2f' % lo[len(COLORS) + i] for i in range(len(SIZES))),
                 ' '.join('%+.2f' % lo[len(COLORS) + len(SIZES) + i] for i in range(len(TEMPS)))))
    Sraw = np.linalg.svd(M, compute_uv=False)
    print('  原始矩阵（未双中心化）前 3 奇异值占比 %s'
          % ' '.join('%.2f' % (s ** 2 / (Sraw ** 2).sum()) for s in Sraw[:3]))

    payload = {
        'phase': 'Q6-B', 'n_obj': len(OBJ), 'n_attr': len(ATTR), 'obj': OBJ, 'attr': ATTR,
        'n_templates': sum(len(v) for v in tmpls.values()),
        'meas_mode': 'single_token_next_logp',
        'C1_color': cg, 'C1_size': sg, 'C2_template_spearman': None if np.isnan(c2_med) else round(c2_med, 4),
        'C3_contrast': round(c3, 4), 'reliable': bool(reliable),
        'r2_curve': {int(k): round(v, 4) for k, v in r2_true.items()},
        'r2_null_curve': {int(k): round(v, 4) for k, v in r2_null.items()},
        'R_best': Rbest, 'R_star': Rstar, 'Q1_verdict': q1,
        'dec_profile': [round(float(x), 4) for x in dec_p],
        'dec_adj_profile': [round(float(x), 4) for x in dec_null_p],
        'dec_diff_profile': [round(float(x), 4) for x in dif],
        'dec_frac_pos': round(f_pos_d, 4), 'dec_frac_neg': round(f_neg_d, 4),
        'Q2_verdict': q2,
        'var_frac': {'row': round(float(v_row / vt), 4), 'col': round(float(v_col / vt), 4),
                     'interaction': round(float(v_int / vt), 4)},
        'peak_ratio_median': round(float(np.median(peak)), 4),
        'n_above_peak_minus1_median': float(np.median(n_above)),
        'n_above_peak_minus1_mean': round(float(np.mean(n_above)), 3),
        'sv_share_top3_interaction': [round(float(s ** 2 / (S ** 2).sum()), 4) for s in S[:3]],
        'matrix': [[round(float(x), 3) for x in row] for row in M],
        'elapsed_s': round(time.time() - t0, 1),
    }
    out = provenance.write_result('pat_q6b', payload, model_path=MODEL_PATH,
                                  seed=SEED, tool='tools/pat_q6b.py')
    print('\n落盘 %s' % out)
    print('Q1=%s  Q2=%s  用时 %.0fs' % (q1, q2, time.time() - t0))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
