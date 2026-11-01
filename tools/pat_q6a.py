# -*- coding: utf-8 -*-
"""Q6-A：关系方向是否被复用（within / A1b / A1p 置换 / 层级逐链 / 留出类比）。

预注册：research/gpt5/docs/PAT_Q6A_PREREG_v1.md（§2 原判据 + §6 跑前修订；阈值不得改）
方案：research/gpt5/docs/PAT_Q6_ATLAS_PLAN_v1.md

三个竞争解释（判决集 §6.6 互斥且穷尽）
  A 冗余复制  → within ≈ across ≈ 0、A1p 不显著        ⇒ NO_STRUCTURE
  C 相对编码  → A1b 成立且 A1p 显著、族间 margin 大     ⇒ RELATION_SPECIFIC
  目标锁定    → within 高但只在同目标配对里成立         ⇒ TARGET_LOCKED
  通用箭头    → A1b 成立但 margin < 0.10（不分关系）    ⇒ GENERIC_ONLY

关键方法（本轮全部难点）
------------------------
A1p 置换检验：族内打乱源—目标配对（**保持目标词多重集不变**），
观察到与零分布共用同一估计器 ⇒ 目标词相似性、‖Δ‖、n 全被吸收。
这是唯一能挡住「F4 只有小/大两个目标、两者本身很像」这类伪影的检验（§6.7）。

A1b：同族内**目标词不同**的配对（挡「指向同一目标词」）。
target_self_cos：族内目标词两两 cos 中位（>0.60 的族 A1b 须打折）。

D26：A3 的 bootstrap 必须与点估计同口径（逐实例按所属折的 l* 评估），
     并断言 CI 包住点估计；否则判 ESTIMATOR_BUG，不输出判决。

用法
----
  SMOKE=1 .venv/Scripts/python.exe -X utf8 tools/pat_q6a.py
  .venv/Scripts/python.exe -X utf8 tools/pat_q6a.py
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

MODEL_PATH = 'D:/develop/model/Qwen/Qwen3-4B'
SEED = 20261101
SMOKE = os.environ.get('SMOKE', '') == '1'
N_BOOT = 100 if SMOKE else 1000
N_PERM = 200
N_FOLD = 5
TMPL = ['我想到一个词：{W}。', '他刚刚提到了{W}。', '我想说的是{W}。']
SLOT = '{W}'

# ---------------- 材料（预注册 §6.5，一字不改） ----------------
FAM = {
    'F1': [('苹果', '水果'), ('香蕉', '水果'), ('桃子', '水果'), ('西瓜', '水果'),
           ('梨', '水果'), ('白菜', '蔬菜'), ('土豆', '蔬菜'), ('黄瓜', '蔬菜'),
           ('萝卜', '蔬菜'), ('狗', '动物'), ('猫', '动物'), ('马', '动物'),
           ('鸟', '动物'), ('鱼', '动物')],
    'F2': [('水果', '食物'), ('蔬菜', '食物'), ('米饭', '食物'), ('面包', '食物'),
           ('鸡蛋', '食物'), ('牛肉', '食物'), ('牛奶', '食物'), ('豆腐', '食物')],
    'F3': [('苹果', '红色'), ('香蕉', '黄色'), ('草', '绿色'), ('雪', '白色'),
           ('煤', '黑色'), ('天空', '蓝色'), ('葡萄', '紫色'), ('橘子', '橙色')],
    'F4': [('蚂蚁', '小'), ('老鼠', '小'), ('西瓜', '大'), ('鲸鱼', '大'),
           ('钉子', '小'), ('蚊子', '小'), ('老虎', '大'), ('楼房', '大')],
    'F5': [('大', '小'), ('高', '矮'), ('快', '慢'), ('冷', '热'),
           ('亮', '暗'), ('多', '少'), ('长', '短'), ('胖', '瘦')],
    'F6': [('刀', '切'), ('笔', '写'), ('钥匙', '开'), ('眼镜', '看'),
           ('杯子', '装'), ('剪刀', '剪'), ('尺子', '量'), ('锤子', '砸')],
}
CHAINS = [('苹果', '水果', '食物'), ('香蕉', '水果', '食物'), ('西瓜', '水果', '食物'),
          ('萝卜', '蔬菜', '食物'), ('土豆', '蔬菜', '食物'), ('黄瓜', '蔬菜', '食物'),
          ('狗', '动物', '生物'), ('猫', '动物', '生物'), ('鸟', '动物', '生物'),
          ('马', '动物', '生物')]
LABEL = {'F1': '类别L1(具体→类)', 'F2': '类别L2(类→上位)', 'F3': '颜色',
         'F4': '大小', 'F5': '反义', 'F6': '工具用途(名→动)'}
MAIN = ['F1', 'F3', 'F4', 'F5']      # 主判据族（F2 单目标不计入，F6 单列）

TH_W, TH_M, TH_B = 0.30, 0.20, 0.30     # within / margin / A1b 阈值（冻结）
TH_FRAC = 0.30                          # 层占比门槛（§6.3/§6.6）
TH_P = 0.05                             # 置换显著性
TH_SELF = 0.60                          # target_self_cos 打折线


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def cos(a, b):
    return float(np.dot(unit(a), unit(b)))


def stat_single(d):
    u = unit(d)
    return float(np.mean(u @ unit(u.mean(0))))


def stat_batch(D):
    U = D / (np.linalg.norm(D, axis=-1, keepdims=True) + 1e-12)
    R = U.mean(1)
    Rn = R / (np.linalg.norm(R, axis=-1, keepdims=True) + 1e-12)
    return np.einsum('pnh,ph->p', U, Rn) / D.shape[1]


def word_span(tok, full, base, c0, c1):
    e = tok(full, return_offsets_mapping=True, add_special_tokens=False)
    offs = e['offset_mapping']
    a0, a1 = c0 + base, c1 + base
    return [i for i, (x, y) in enumerate(offs) if x < a1 and y > a0]


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    rng = np.random.default_rng(SEED)
    dev = 'cuda'

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, attn_implementation='eager').to(dev).eval()
    L = model.config.num_hidden_layers
    H = model.config.hidden_size
    NL = L + 1

    vocab = []
    for pairs in FAM.values():
        for a, b in pairs:
            for w in (a, b):
                if w not in vocab:
                    vocab.append(w)
    for ch in CHAINS:
        for w in ch:
            if w not in vocab:
                vocab.append(w)
    widx = {w: i for i, w in enumerate(vocab)}
    print('词表 %d、族 %d、实例 %d、链 %d'
          % (len(vocab), len(FAM), sum(len(v) for v in FAM.values()), len(CHAINS)))

    tmpl = TMPL[:1] if SMOKE else TMPL
    V = np.zeros((len(vocab), NL, H), dtype=np.float32)
    ntoks, nfail = {}, 0
    with torch.no_grad():
        for wi, w in enumerate(vocab):
            acc = np.zeros((NL, H), dtype=np.float32)
            got = 0
            for t in tmpl:
                user = t.replace(SLOT, w)
                full = render(tok, user)
                base = full.find(user)
                if base < 0:
                    continue
                c0 = t.index(SLOT)
                sp = word_span(tok, full, base, c0, c0 + len(w))
                if not sp:
                    continue
                ids = tok(full, add_special_tokens=False)['input_ids']
                out = model(input_ids=torch.tensor([ids], device=dev),
                            use_cache=False, output_hidden_states=True)
                for li in range(NL):
                    acc[li] += out.hidden_states[li][0, sp, :].float().mean(0).cpu().numpy()
                got += 1
                ntoks[w] = len(sp)
            V[wi] = acc / got if got else 0.0
            nfail += (got == 0)
    multi = {w: n for w, n in ntoks.items() if n > 1}
    print('表征 %.0fs，失败 %d；多 token 词 %d/%d，例：%s'
          % (time.time() - t0, nfail, len(multi), len(ntoks), list(multi.items())[:6]))

    inst = [(f, widx[a], widx[b]) for f, ps in FAM.items() for a, b in ps
            if a in widx and b in widx]
    print('可用实例 %d' % len(inst))

    # ---------- 逐层：within / A1b / across / A1p / target_self ----------
    W_th = {f: np.zeros(NL) for f in FAM}
    W_b = {f: np.zeros(NL) for f in FAM}
    W_ac = {f: np.zeros(NL) for f in FAM}
    P_p = {f: np.full(NL, np.nan) for f in FAM}
    S_self = {f: np.full(NL, np.nan) for f in FAM}
    for li in range(NL):
        D = unit(np.stack([V[b, li] - V[a, li] for _, a, b in inst]))
        for f in FAM:
            ii = [k for k, x in enumerate(inst) if x[0] == f]
            oth = [k for k, x in enumerate(inst) if x[0] != f]
            allp = [cos(D[i], D[j]) for x, i in enumerate(ii) for j in ii[x + 1:]]
            W_th[f][li] = float(np.median(allp)) if allp else np.nan
            bp = [cos(D[i], D[j]) for x, i in enumerate(ii) for j in ii[x + 1:]
                  if inst[i][2] != inst[j][2]]
            W_b[f][li] = float(np.median(bp)) if bp else np.nan
            ac = []
            for i in ii:
                wi_ = {inst[i][1], inst[i][2]}
                for j in oth:
                    if wi_ & {inst[j][1], inst[j][2]}:
                        continue
                    ac.append(cos(D[i], D[j]))
            W_ac[f][li] = float(np.median(ac)) if ac else np.nan
            # target_self_cos
            tg = sorted({inst[k][2] for k in ii})
            if len(tg) >= 2:
                S_self[f][li] = float(np.median([cos(V[x, li], V[y, li])
                                                 for i_, x in enumerate(tg)
                                                 for y in tg[i_ + 1:]]))
            # A1p 置换
            if len(ii) >= 4:
                A = V[[inst[k][1] for k in ii], li]
                B = V[[inst[k][2] for k in ii], li]
                obs = stat_single(B - A)
                perms = np.stack([rng.permutation(len(ii)) for _ in range(N_PERM)])
                null = stat_batch(B[perms] - A[None, :, :])
                P_p[f][li] = float(np.mean(null >= obs))

    print('\n=== A1 全层剖面（D25：37 层全打印）===')
    for f in FAM:
        print('%s %-14s within%s' % (f, LABEL[f], ' '.join('%.2f' % x for x in W_th[f])))
        print('%s %-14s A1b  %s' % (' ' * len(f), '', ' '.join('%.2f' % x for x in W_b[f])))
        print('%s %-14s margin%s' % (' ' * len(f), '',
                                     ' '.join('%+.2f' % x for x in (W_th[f] - W_ac[f]))))
        print('%s %-14s p_perm%s' % (' ' * len(f), '',
                                     ' '.join('--' if np.isnan(x) else ('!!' if x < TH_P else '%2d' % int(x * 100))
                                              for x in P_p[f])))

    mgn = {f: W_th[f] - W_ac[f] for f in FAM}
    fr = lambda c: float(np.mean(c))
    fr_w = {f: fr((W_th[f] >= TH_W) & (mgn[f] >= TH_M)) for f in FAM}
    fr_b = {f: fr((W_b[f] >= TH_B) & (mgn[f] >= TH_M)) for f in FAM}
    fr_lowm = {f: fr(mgn[f] < 0.10) for f in FAM}
    fr_p = {f: fr(np.nan_to_num(P_p[f], nan=1.0) < TH_P) for f in FAM}

    print('\n=== A1 层占比（达阈层数/37；p<0.05 记 !!）===')
    for f in FAM:
        print('%-3s %-14s within %2d  A1b %2d  置换显著 %2d  margin<0.10 %2d  '
              'target_self %.2f  n=%d'
              % (f, LABEL[f], round(fr_w[f] * NL), round(fr_b[f] * NL),
                 round(fr_p[f] * NL), round(fr_lowm[f] * NL),
                 float(np.nanmedian(S_self[f])), len(FAM[f])))

    confounded = [f for f in FAM if fr_b[f] >= TH_FRAC and fr_p[f] < 0.10]
    spec = sum(1 for f in MAIN if fr_b[f] >= TH_FRAC and f not in confounded)
    lock = sum(1 for f in MAIN if fr_b[f] < 0.10)
    gen = sum(1 for f in MAIN if fr_b[f] >= TH_FRAC and f not in confounded
              and fr_lowm[f] >= 0.70)
    nogeneric = sum(1 for f in MAIN if fr_w[f] < 0.10)
    nostr = sum(1 for f in FAM if fr_w[f] < 0.10)
    if spec >= 3:
        A1 = 'RELATION_SPECIFIC'
    elif lock >= 3:
        A1 = 'TARGET_LOCKED'
    elif gen >= 3:
        A1 = 'GENERIC_ONLY'
    elif nostr >= 4:
        A1 = 'NO_STRUCTURE'
    else:
        A1 = 'MIXED'
    print('\nA1 判决 = %s（A1b 有效 %d/4、A1b 近零 %d/4、generic %d/4、'
          'within 近零 %d/6、A1p 剔除 %s）'
          % (A1, spec, lock, gen, nogeneric, confounded or '无'))

    # bootstrap margin（峰值层，仅作功效说明）
    lpeak = int(np.nanargmax([np.mean([mgn[f][l] for f in MAIN]) for l in range(NL)]))

    def boot_margin(f, B=N_BOOT):
        D = unit(np.stack([V[b, lpeak] - V[a, lpeak] for _, a, b in inst]))
        ii = [k for k, x in enumerate(inst) if x[0] == f]
        oth = [k for k, x in enumerate(inst) if x[0] != f]
        vals = []
        for _ in range(B):
            pick = rng.choice(ii, len(ii), replace=True)
            w = [cos(D[i], D[j]) for x, i in enumerate(pick) for j in pick[x + 1:]]
            ac = [cos(D[i], D[j]) for i in pick for j in oth
                  if not ({inst[i][1], inst[i][2]} & {inst[j][1], inst[j][2]})]
            if len(w) > 1 and ac:
                vals.append(np.median(w) - np.median(ac))
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) \
            if vals else (float('nan'), float('nan'))

    # 族×族：若「跨词区转换」是全部原因，F3-F4（同为名→修饰）应高
    fl = list(FAM)
    Fmat = {}
    for f1 in fl:
        for f2 in fl:
            if f1 >= f2:
                continue
            i1 = [k for k, x in enumerate(inst) if x[0] == f1]
            i2 = [k for k, x in enumerate(inst) if x[0] == f2]
            vals = []
            for i in i1:
                for j in i2:
                    if {inst[i][1], inst[i][2]} & {inst[j][1], inst[j][2]}:
                        continue
                    vals.append(cos(V[inst[i][2], lpeak] - V[inst[i][1], lpeak],
                                    V[inst[j][2], lpeak] - V[inst[j][1], lpeak]))
            Fmat['%s-%s' % (f1, f2)] = float(np.median(vals)) if vals else float('nan')
    print('\n族×族 Δ 中位余弦 @L%d（检验跨词区解释）：' % lpeak)
    for k, v in Fmat.items():
        print('   %-8s %+.3f' % (k, v))

    ci = {f: boot_margin(f) for f in FAM}
    print('\n峰值层 L%d（仅供描述，不用于判决）' % lpeak)
    for f in FAM:
        print('  %-3s within %.3f  A1b %.3f  across %.3f  margin %+.3f  CI [%.2f,%.2f]  '
              'p_perm %.3f  target_self %.2f'
              % (f, W_th[f][lpeak], W_b[f][lpeak], W_ac[f][lpeak], mgn[f][lpeak],
                 ci[f][0], ci[f][1], P_p[f][lpeak], S_self[f][lpeak]))

    # ---------- A2：逐链（同一支抽象箭头能否连用两次） ----------
    Dc = {ch: (widx[ch[0]], widx[ch[1]], widx[ch[2]]) for ch in CHAINS
          if all(w in widx for w in ch)}
    prof_h, prof_c = np.zeros(NL), np.zeros(NL)
    for li in range(NL):
        h1 = {ch: V[ix[1], li] - V[ix[0], li] for ch, ix in Dc.items()}
        h2 = {ch: V[ix[2], li] - V[ix[1], li] for ch, ix in Dc.items()}
        ks = list(Dc)
        prof_h[li] = float(np.median([cos(h1[k], h2[k]) for k in ks]))
        prof_c[li] = float(np.median([cos(h1[ks[i]], h2[ks[j]]) for i in range(len(ks))
                                      for j in range(len(ks)) if i != j]))
    print('\n=== A2 逐链（D25 全层）===')
    print('同链 cos(h1,h2) %s' % ' '.join('%.2f' % x for x in prof_h))
    print('跨链 对照       %s' % ' '.join('%.2f' % x for x in prof_c))
    fr_h = fr((prof_h >= TH_W) & ((prof_h - prof_c) >= 0.10))
    print('A2 达阈层 %d/37；峰值 %.3f @L%d（对照 %.3f）'
          % (round(fr_h * NL), float(prof_h.max()), int(prof_h.argmax()),
             float(prof_c[int(prof_h.argmax())])))
    A2 = ('HIERARCHY_SHARED' if fr_h >= TH_FRAC else
          'HIERARCHY_NOT_SHARED' if float(prof_h.max()) < 0.10 else 'HIERARCHY_WEAK')
    print('A2 判决 = %s（中位 %.3f）' % (A2, float(np.median(prof_h))))

    # ---------- A2b：径向编码（A2 为负时的正向替代，§6.9） ----------
    prof_r = np.zeros(NL)
    ch_ok = {ch: True for ch in Dc}
    for li in range(NL):
        cvec = unit(V[:, li].mean(0))
        cen = V[:, li] @ cvec
        ok = [cen[ix[0]] < cen[ix[1]] < cen[ix[2]] for ix in Dc.values()]
        prof_r[li] = float(np.mean(ok))
    fr_r = fr(prof_r >= 0.70)
    print('A2b 单调链占比 %s' % ' '.join('%.2f' % x for x in prof_r))
    print('A2b 达阈层 %d/37（峰值 %.2f @L%d）'
          % (round(fr_r * NL), float(prof_r.max()), int(prof_r.argmax())))
    A2b = 'HIERARCHY_RADIAL' if (fr_r >= TH_FRAC and float(prof_r.max()) >= 0.70) \
        else 'RADIAL_NOT_SUPPORTED'
    print('A2b 判决 = %s（峰值层链占比 %.2f）' % (A2b, float(prof_r[int(prof_r.argmax())])))

    # ---------- A3：留出类比（口径统一见 §6.8 / D26） ----------
    cand = np.arange(len(vocab))

    def fold_eval(li, test_tgt, allow):
        U = unit(V[:, li])
        out = []
        for k, (f, a, b) in enumerate(inst):
            if b not in test_tgt:
                continue
            ii = [j for j, x in enumerate(inst) if x[0] == f and j != k
                  and inst[j][1] in allow and inst[j][2] in allow]
            if not ii:
                continue
            r = np.mean([V[inst[j][2], li] - V[inst[j][1], li] for j in ii], axis=0)
            cc = cand[cand != a]
            out.append((k, int(cc[int((U[cc] @ (V[a, li] + r)).argmax())] == b),
                        int(cc[int((U[cc] @ V[a, li]).argmax())] == b)))
        return out

    words = np.arange(len(vocab))
    rng.shuffle(words)
    folds = [set(words[i::N_FOLD].tolist()) for i in range(N_FOLD)]
    recs, ls3 = [], []
    for fi in range(N_FOLD):
        te = folds[fi]
        tr = set(range(len(vocab))) - te
        best, bl = -1.0, 0
        for li in range(NL):
            ev = fold_eval(li, tr, tr)
            if ev:
                a_ = np.mean([e[1] for e in ev])
                if a_ > best:
                    best, bl = a_, li
        ls3.append(bl)
        recs += fold_eval(bl, te, tr)      # 与点估计同口径：该折自己选的 l*
    a3 = float(np.mean([r[1] for r in recs])) if recs else float('nan')
    nn3 = float(np.mean([r[2] for r in recs])) if recs else float('nan')
    print('\nA3 每折 l* = %s；留出 acc %.3f，NN 基线 %.3f，n=%d' % (ls3, a3, nn3, len(recs)))

    maj_hit = maj_n = 0
    for f, ps in FAM.items():
        tg = [b for _, b in ps]
        maj_hit += tg.count(max(set(tg), key=tg.count))
        maj_n += len(tg)
    maj = maj_hit / maj_n
    rnd = 1.0 / (len(vocab) - 1)
    print('多数基线 %.3f，随机 %.4f' % (maj, rnd))

    bo = []
    for _ in range(N_BOOT):
        pick = rng.choice(len(recs), len(recs), replace=True)
        bo.append(float(np.mean([recs[i][1] for i in pick])))
    a3_ci = (float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5)))
    ok_ci = a3_ci[0] <= a3 <= a3_ci[1]
    print('A3 CI [%.3f, %.3f]  包住点估计=%s' % (a3_ci[0], a3_ci[1], ok_ci))
    if not ok_ci:
        print('!! ESTIMATOR_BUG：CI 不含点估计，先修估计器（D26）')
        A3 = 'ESTIMATOR_BUG'
    else:
        A3 = ('PREDICTIVE' if (a3 >= 0.50 and a3 >= maj + 0.15 and a3 >= nn3 + 0.15)
              else 'NOT_PREDICTIVE')
    print('A3 判决 = %s' % A3)
    pf = {f: [0, 0] for f in FAM}
    for k, h, _ in recs:
        pf[inst[k][0]][1] += 1
        pf[inst[k][0]][0] += h
    print('A3 逐族（留出）：%s' % '  '.join(
        '%s %d/%d' % (f, pf[f][0], pf[f][1]) for f in FAM if pf[f][1]))

    norms = {f: float(np.median([np.linalg.norm(V[b, lpeak] - V[a, lpeak])
                                 for x, a, b in inst if x == f])) for f in FAM}
    print('\n‖Δ‖ 中位 @L%d：%s' % (lpeak, {f: round(v, 1) for f, v in norms.items()}))

    payload = {
        'phase': 'Q6-A', 'n_words': len(vocab), 'n_inst': len(inst),
        'n_templates': len(tmpl), 'n_perm': N_PERM,
        'peak_layer_descriptive_only': lpeak,
        'A1_verdict': A1, 'A1_confounded_families': confounded,
        'A1_frac_within': {f: round(fr_w[f], 4) for f in FAM},
        'A1_frac_A1b': {f: round(fr_b[f], 4) for f in FAM},
        'A1_frac_perm_sig': {f: round(fr_p[f], 4) for f in FAM},
        'A1_frac_low_margin': {f: round(fr_lowm[f], 4) for f in FAM},
        'A1_profile_within': {f: [round(float(x), 4) for x in W_th[f]] for f in FAM},
        'A1_profile_A1b': {f: [None if np.isnan(x) else round(float(x), 4)
                               for x in W_b[f]] for f in FAM},
        'A1_profile_across': {f: [round(float(x), 4) for x in W_ac[f]] for f in FAM},
        'A1_profile_perm_p': {f: [None if np.isnan(x) else round(float(x), 4)
                                  for x in P_p[f]] for f in FAM},
        'target_self_cos_median': {f: (None if np.all(np.isnan(S_self[f]))
                                       else round(float(np.nanmedian(S_self[f])), 4))
                                   for f in FAM},
        'A1_peak_values': {f: {'within': round(float(W_th[f][lpeak]), 4),
                               'A1b': None if np.isnan(W_b[f][lpeak])
                               else round(float(W_b[f][lpeak]), 4),
                               'across': round(float(W_ac[f][lpeak]), 4),
                               'margin': round(float(mgn[f][lpeak]), 4),
                               'margin_CI': [round(ci[f][0], 4), round(ci[f][1], 4)],
                               'perm_p': None if np.isnan(P_p[f][lpeak])
                               else round(float(P_p[f][lpeak]), 4),
                               'target_self_cos': None if np.isnan(S_self[f][lpeak])
                               else round(float(S_self[f][lpeak]), 4),
                               'n': len(FAM[f])} for f in FAM},
        'A1_fam_cross_matrix_at_peak': {k: round(v, 4) for k, v in Fmat.items()},
        'A1_counts': {'A1b_valid': spec, 'A1b_near_zero': lock, 'generic': gen,
                      'within_near_zero_main': nogeneric, 'within_near_zero_all': nostr},
        'A2_profile_chain': [round(float(x), 4) for x in prof_h],
        'A2_profile_cross': [round(float(x), 4) for x in prof_c],
        'A2_frac': round(fr_h, 4), 'A2_median': round(float(np.median(prof_h)), 4),
        'A2_verdict': A2,
        'A2b_profile_monotone_frac': [round(float(x), 4) for x in prof_r],
        'A2b_frac': round(fr_r, 4),
        'A2b_peak': round(float(prof_r.max()), 4),
        'A2b_verdict': A2b,
        'A3_holdout_acc': round(a3, 4), 'A3_nn_baseline': round(nn3, 4),
        'A3_majority_baseline': round(maj, 4), 'A3_random': round(rnd, 4),
        'A3_CI': [round(a3_ci[0], 4), round(a3_ci[1], 4)],
        'A3_ci_contains_point': bool(ok_ci),
        'A3_lstar_per_fold': ls3, 'A3_n_eval': len(recs),
        'A3_per_family': {f: [pf[f][0], pf[f][1]] for f in FAM if pf[f][1]},
        'A3_verdict': A3,
        'delta_norm_median': {f: round(v, 2) for f, v in norms.items()},
        'token_len': ntoks,
        'elapsed_s': round(time.time() - t0, 1),
    }
    out = provenance.write_result('pat_q6a', payload, model_path=MODEL_PATH,
                                  seed=SEED, tool='tools/pat_q6a.py')
    print('\n落盘 %s' % out)
    print('A1=%s  A2=%s  A2b=%s  A3=%s  用时 %.0fs'
          % (A1, A2, A2b, A3, time.time() - t0))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
