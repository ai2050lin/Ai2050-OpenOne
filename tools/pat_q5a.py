# -*- coding: utf-8 -*-
"""Q5-A：融合的形状——赢家通吃还是加权？

预注册：research/gpt5/docs/PAT_Q5A_PREREG_v1.md（判据跑前冻结）
框架：PSF-3 / 代数签名 S 位

为什么这一测不会被切片救
------------------------
判据是**分布形状**（熵、单点质量、头一致性），不是"某处显著"。
换对照只改幅度不改形状，所以它自带备择、无法用细分换结果（D24 最高优先级）。

三个量各测一个侧面
------------------
  h3      三槽归一化熵（0 独占 / 1 均分）
  maxpos  全句最大单点质量（0 均匀 / 1 所有头钉死一位）
  unanim  头间 argmax 一致性（1 全头同向 / 低=维度分工）

unanim 是“多维度”假设的直接检验：Q4-R 只看了**跨头求和后**的 argmax，
从未看逐头。若各头分工不同维度，它们不应全部指向同一槽。

冲突臂
------
真双查询句 `ctx + '是谁送的？送给了谁？'` vs **等长非查询对照**
`ctx + '是谁送的？随便说说。'`。用差值，防长度混杂（RoPE 相位随绝对位置变）。

用法
----
  SMOKE=1 .venv/Scripts/python.exe -X utf8 tools/pat_q5a.py
  .venv/Scripts/python.exe -X utf8 tools/pat_q5a.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / 'tools')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import provenance
from pat_q1_unbind import PERSONS, build, render

MODEL_PATH = 'D:/develop/model/Qwen/Qwen3-4B'
SEED = 20261101
SMOKE = os.environ.get('SMOKE', '') == '1'
N_TRIP = 6 if SMOKE else 24

NOUNS = ['苹果', '香蕉', '桃子', '面包', '蛋糕', '杯子', '碗', '毛巾', '雨伞', '铅笔',
         '书包', '钥匙', '手表', '眼镜', '帽子', '围巾', '手套', '鞋子', '镜子', '想法',
         '计划', '建议', '问题', '答案', '理由', '办法', '习惯', '故事', '照片', '礼物']

QUERIES = {'ag': '是谁送的？', 're': '送给了谁？', 'th': '送的是什么？'}
DUAL = '是谁送的？送给了谁？'          # 真双查询
DUAL_CTRL = '是谁送的？随便说说。'      # 等长非查询对照（同长度量级）
ROLES = ('agent', 'recipient', 'theme')
FORMS = ('A', 'C')
KEY_LAYERS = (21, 22, 23, 28)


def rope_vec(v, cos_t, sin_t):
    hd = v.shape[-1] // 2
    x1, x2 = v[..., :hd], v[..., hd:]
    return v * cos_t + torch.cat([-x2, x1], dim=-1) * sin_t


def slot_spans(tok, full, ranges, base):
    e = tok(full, return_offsets_mapping=True, add_special_tokens=False)
    offs, ids = e['offset_mapping'], e['input_ids']
    out = {}
    for role, (c0, c1) in ranges.items():
        a0, a1 = c0 + base, c1 + base
        out[role] = [i for i, (x, y) in enumerate(offs) if x < a1 and y > a0]
    return out, ids


def entropy(p):
    p = np.asarray(p, dtype=np.float64)
    p = p / max(p.sum(), 1e-12)
    q = p[p > 0]
    return float(-(q * np.log(q)).sum())


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    dev = 'cuda'

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, attn_implementation='eager').to(dev).eval()
    cfg = model.config
    L, QH, KV = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
    HD = getattr(cfg, 'head_dim', 128)
    rep, scale = QH // KV, HD ** -0.5

    cap = {}

    def mk(li, name):
        def hook(mod, inp, out):
            cap.setdefault(li, {})[name] = out.detach()
        return hook

    for li in range(L):
        att = model.model.layers[li].self_attn
        att.q_norm.register_forward_hook(mk(li, 'q'))
        att.k_norm.register_forward_hook(mk(li, 'k'))
    model.model.rotary_emb.register_forward_hook(lambda m, i, o: cap.__setitem__('rope', o))

    def fwd(ids):
        with torch.no_grad():
            cap.clear()
            return model(input_ids=torch.tensor([ids], device=dev), use_cache=False)

    def attn_metrics(sp, sel_layers):
        """返回 {li: dict(h3, maxpos, unanim, cover, span_conc)}。"""
        cos, sin = cap['rope']
        cl, sl = cos[0, -1].float(), sin[0, -1].float()
        cT, sT = cos[0].float(), sin[0].float()
        out = {}
        for li in sel_layers:
            qr = rope_vec(cap[li]['q'][0, -1].float(), cl, sl)
            kr = rope_vec(cap[li]['k'][0].float(), cT[:, None, :], sT[:, None, :])
            Ac = torch.softmax(
                torch.einsum('hd,thd->ht', qr, kr.repeat_interleave(rep, dim=1)) * scale,
                dim=1).cpu().numpy()                     # [QH, T]
            # 逐头的三槽质量
            ph = np.stack([Ac[:, sp[r]].sum(1) for r in ROLES], axis=1)   # [QH, 3]
            tot = ph.sum(1)                                # [QH] 每头在三槽上的总质量
            # 头间一致性：逐头 argmax 的众数占比（只在三槽都非零时算）
            am = ph.argmax(1)
            vals, cnts = np.unique(am, return_counts=True)
            unanim = float(cnts.max() / len(am))
            # 跨头聚合的三槽分布
            p3 = ph.mean(0)
            p3n = p3 / max(p3.sum(), 1e-12)
            # 全句最大单点质量
            mp = float(Ac.mean(0).max())
            # 三槽总覆盖
            cover = float(Ac.mean(0)[[i for r in ROLES for i in sp[r]]].sum())
            # 跨度内集中度：多 token 槽把质量放在首个 token 的比例
            conc = []
            for r in ROLES:
                if len(sp[r]) > 1:
                    seg = Ac.mean(0)[sp[r]]
                    conc.append(float(seg.max() / max(seg.sum(), 1e-12)))
            am_pos = Ac.mean(0)                       # [T] 跨头平均
            tp = np.argsort(-am_pos)[:5]
            out[li] = {'h3': entropy(p3n) / np.log(3), 'maxpos': mp, 'unanim': unanim,
                       'cover': cover, 'span_conc': float(np.mean(conc)) if conc else 1.0,
                       'sink0': float(am_pos[0]), 'sink1': float(am_pos[1]),
                       'ph': ph.copy(), 'ph_tot': tot.copy(),
                       'sink_h': Ac[:, 0].copy(),
                       'topk': [[int(x), float(am_pos[x])] for x in tp],
                       'p3': p3n.tolist()}
        return out

    pw = PERSONS if not SMOKE else PERSONS[:4]
    nw = NOUNS if not SMOKE else NOUNS[:6]
    trips = [(pw[i % len(pw)], pw[(i + 3) % len(pw)], nw[i % len(nw)])
             for i in range(N_TRIP)]
    print('三元组 %d' % len(trips))

    sel = list(range(L))
    per_cell = []     # 每格 {qk, form, li: metrics}
    beh = {q: [0, 0] for q in QUERIES}

    # ---------- Part 1：144 格单查询 ----------
    for ti, trip in enumerate(trips):
        for form in FORMS:
            ctx, rg = build(trip, form)
            for qk, qtext in QUERIES.items():
                user = ctx + qtext
                full = render(tok, user)
                base = full.find(user)
                sp, ids = slot_spans(tok, full, rg, base)
                if any(not sp[r] for r in ROLES):
                    continue
                out = fwd(ids)
                lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
                sc = {x: float(lp[tok(x, add_special_tokens=False)['input_ids'][0]])
                      for x in trip}
                want = {'ag': trip[0], 're': trip[1], 'th': trip[2]}[qk]
                beh[qk][1] += 1
                beh[qk][0] += int(max(sc, key=sc.get) == want)
                per_cell.append({'ti': ti, 'form': form, 'qk': qk, 'ids': ids,
                                 'sp': {r: list(v) for r, v in sp.items()},
                                 'm': attn_metrics(sp, sel)})
    N = len(per_cell)
    print('单查询格 %d，行为门 %s，%.0fs'
          % (N, {q: '%d/%d' % tuple(beh[q]) for q in QUERIES}, time.time() - t0))

    # ---------- Part 2：冲突臂 48 × 2 ----------
    dual = []
    for ti, trip in enumerate(trips):
        for form in FORMS:
            ctx, rg = build(trip, form)
            row = {}
            for tag, tail in (('dual', DUAL), ('ctrl', DUAL_CTRL)):
                user = ctx + tail
                full = render(tok, user)
                base = full.find(user)
                sp, ids = slot_spans(tok, full, rg, base)
                if any(not sp[r] for r in ROLES):
                    continue
                fwd(ids)
                row[tag] = attn_metrics(sp, list(KEY_LAYERS))
            if 'dual' in row and 'ctrl' in row:
                dual.append(row)
    print('冲突臂 %d 对，%.0fs' % (len(dual), time.time() - t0))

    # ---------- 汇总 ----------
    def collect(metric, layers):
        v = []
        for c in per_cell:
            for li in layers:
                v.append(c['m'][li][metric])
        return np.array(v, dtype=np.float64)

    key_vals = {k: collect(k, KEY_LAYERS) for k in ('h3', 'maxpos', 'unanim', 'cover', 'span_conc')}
    med = {k: float(np.median(v)) for k, v in key_vals.items()}
    q1, q3 = {}, {}
    for k, v in key_vals.items():
        q1[k], q3[k] = float(np.percentile(v, 25)), float(np.percentile(v, 75))

    print('\n关键层 %s（%d 个观测）' % (KEY_LAYERS, len(key_vals['h3'])))
    for k in ('h3', 'maxpos', 'unanim', 'cover', 'span_conc'):
        print('  %-10s 中位 %.3f  IQR [%.3f, %.3f]  min %.3f max %.3f'
              % (k, med[k], q1[k], q3[k], key_vals[k].min(), key_vals[k].max()))

    # D5：先看分布形状，双峰则禁报均值
    def bimodality(v):
        v = np.sort(v)
        h, edges = np.histogram(v, bins=10, range=(0, 1))
        peaks = sum(1 for i in range(1, 9) if h[i] > h[i - 1] and h[i] > h[i + 1])
        return peaks, h.tolist()

    pk, hist = bimodality(key_vals['h3'])
    print('  h3 直方图（10 桶 0..1）%s  峰数 %d' % (hist, pk))

    # 逐层剖面（D25：全部层）
    prof = {}
    for k in ('h3', 'maxpos', 'unanim', 'cover'):
        prof[k] = [round(float(np.median([c['m'][li][k] for c in per_cell])), 4)
                   for li in range(L)]
    print('\n逐层剖面（中位）：')
    for k in ('h3', 'maxpos', 'unanim', 'cover'):
        print('  %-8s ' % k + ' '.join('%.2f' % x for x in prof[k]))

    # ---- 头级分工诊断：这是“多维度”假设的直接检验 ----
    # 对每个头，在三槽上取 argmax，统计它与“正确答案槽”的一致率（只在关键层）
    ans_role = np.array([{'ag': 0, 're': 1, 'th': 2}[c['qk']] for c in per_cell])
    hitL = np.zeros((len(KEY_LAYERS), N, QH), dtype=np.int8)
    sig = np.zeros((N, QH), dtype=bool)
    for i, c in enumerate(per_cell):
        for j, li in enumerate(KEY_LAYERS):
            m = c['m'][li]
            hitL[j, i] = (m['ph'].argmax(1) == ans_role[i]).astype(np.int8)
            sig[i] |= (m['ph_tot'] >= 0.20)
    head_acc = hitL.sum((1, 2)) / N
    head_acc_L = {li: hitL[j].mean(0) for j, li in enumerate(KEY_LAYERS)}
    n_sig_per_cell = sig.sum(1)
    n_sig_per_head = sig.sum(0)
    print('\n头级分工（%d 格 × %d 层）' % (N, len(KEY_LAYERS)))
    print('  每格“有信号头”数：中位 %d / %d（阈值：该头三槽覆盖≥0.20）'
          % (int(np.median(n_sig_per_cell)), QH))
    top = np.argsort(-head_acc)[:8]
    print('  逐头指对率 top8: %s' % ', '.join('H%d=%.2f(%d格)' % (h, head_acc[h], n_sig_per_head[h])
                                              for h in top))
    print('  逐头指对率 分布: 中位 %.2f  >0.80 的头 %d 个  <0.50 的头 %d 个'
          % (float(np.median(head_acc)), int((head_acc > 0.80).sum()),
             int((head_acc < 0.50).sum())))
    print('  （随机=0.333。若多数头>0.80 ⇒ 32 个头做同一件事、无维度分工；'
          '若少数高 ⇒ 稀疏路由）')
    # 只看有信号的头，重算一致性（避开 sink 头污染）
    uni_sig = []
    for i, c in enumerate(per_cell):
        for li in KEY_LAYERS:
            m = c['m'][li]
            ok = m['ph_tot'] >= 0.20
            if ok.sum() < 4:
                continue
            am = m['ph'][ok].argmax(1)
            _, cnts = np.unique(am, return_counts=True)
            uni_sig.append(cnts.max() / len(am))
    uni_sig = np.array(uni_sig)
    print('  有信号头之间的一致性：中位 %.3f（n=%d 格层）'
          % (float(np.median(uni_sig)), len(uni_sig)))

    # ---- k 曲线：聚合多少个头才够 ----
    rng2 = np.random.default_rng(SEED + 7)
    ks = [1, 2, 4, 8, 16, 32]
    kcurve = {}
    kcurve = {li: {} for li in KEY_LAYERS}
    for j, li in enumerate(KEY_LAYERS):
        phL = np.stack([c['m'][li]['ph'] for c in per_cell])      # [N, QH, 3]
        for k in ks:
            accs = []
            for _ in range(200):
                pick = rng2.choice(QH, k, replace=False)
                agg = phL[:, pick].sum(1)                          # [N, 3]
                accs.append(float((agg.argmax(1) == ans_role).mean()))
            kcurve[li][k] = (round(float(np.mean(accs)), 4), round(float(np.std(accs)), 4))
        print('  L%-2d k 曲线：%s' % (li, '  '.join('k=%d %.3f±%.3f' % (k, *kcurve[li][k])
                                                   for k in ks)))
    print('  逐头指对率（按层）：%s'
          % '  '.join('L%d 中位%.2f max%.2f' % (li, float(np.median(head_acc_L[li])),
                                              float(head_acc_L[li].max())) for li in KEY_LAYERS))

    # ---- sink 是否承担“分维度寄存”：逐头 sink 质量的分散度，及其与指对率的关系 ----
    sk = np.stack([c['m'][28]['sink_h'] for c in per_cell])      # [N, QH]  L28
    h28 = head_acc_L[28]
    sd_head = float(np.mean(sk.std(1)))           # 同一格内头间标准差（分化程度）
    print('  sink 诊断（L28）：逐头 pos0 质量 总体中位 %.3f，格内头间 sd 均值 %.3f'
          % (float(np.median(sk)), sd_head))
    print('    逐头 sink 质量 中位 %.3f  范围 [%.3f, %.3f]；'
          '与逐头指对率的 Pearson r = %.3f'
          % (float(np.median(sk.mean(0))), float(sk.mean(0).min()),
             float(sk.mean(0).max()),
             float(np.corrcoef(sk.mean(0), h28)[0, 1])))
    print('    每头均值：%s' % ' '.join('%.2f' % x for x in sk.mean(0)))

    # ---- 位置诊断：剩下的注意力在哪 ----
    sink0 = np.array([c['m'][li]['sink0'] for c in per_cell for li in KEY_LAYERS])
    sink1 = np.array([c['m'][li]['sink1'] for c in per_cell for li in KEY_LAYERS])
    print('\n位置诊断（关键层）：pos0 质量中位 %.3f [%.3f,%.3f]、pos1 %.3f'
          % (np.median(sink0), np.percentile(sink0, 25), np.percentile(sink0, 75),
             np.median(sink1)))
    # 三槽 + pos0/pos1 覆盖了多少
    print('  三槽 %.3f + pos0+pos1 %.3f = %.3f'
          % (med['cover'], float(np.median(sink0 + sink1)),
             med['cover'] + float(np.median(sink0 + sink1))))
    shown = 0
    for c in per_cell:
        if c['form'] == 'A' and c['qk'] == 'ag' and shown < 2:
            li = 28
            toks = [tok.decode([i]) for i in c['ids']]
            print('  样例 (form=%s q=%s) L28 前五位置: %s'
                  % (c['form'], c['qk'],
                     ', '.join('%d:%r=%.3f' % (pp, toks[pp], mm)
                               for pp, mm in c['m'][li]['topk'])))
            print('       三槽位置 %s，p3=%s'
                  % ({r: c['sp'][r] for r in ROLES},
                     ['%.3f' % x for x in c['m'][li]['p3']]))
            shown += 1

    # 冲突臂差值（用等长非查询对照，不用单查询）
    d_h3 = []
    for row in dual:
        for li in KEY_LAYERS:
            d_h3.append(row['dual'][li]['h3'] - row['ctrl'][li]['h3'])
    d_h3 = np.array(d_h3)
    a5_med = float(np.median(d_h3))
    print('\n冲突臂 h3 差值（双查询 − 等长对照）：中位 %+.3f  IQR [%+.3f, %+.3f]  n=%d'
          % (a5_med, float(np.percentile(d_h3, 25)), float(np.percentile(d_h3, 75)), len(d_h3)))

    # ---------- 判据 ----------
    A1 = bool(med['h3'] <= 0.35)
    A2 = bool(med['maxpos'] >= 0.60)
    A3 = bool(float(np.median(uni_sig)) >= 0.80) if len(uni_sig) else False
    A4 = bool(med['cover'] >= 0.40)
    A5 = bool(a5_med >= 0.30)
    if not A4:
        verdict = 'FRAMEWORK_INVALID'
    elif A1 and A2 and not A5:
        verdict = 'WINNER_TAKE_ALL'
    elif (not A1) and A5:
        verdict = 'WEIGHTED_SUM'
    else:
        verdict = 'MIXED'
    print('\n判决 = %s' % verdict)
    print('A1 h3≤.35 %s | A2 maxpos≥.60 %s | A3 unanim≥.80 %s | A4 cover≥.40 %s | A5 Δh3≥.30 %s'
          % (A1, A2, A3, A4, A5))

    rec = {
        'seed': SEED, 'smoke': SMOKE, 'n_cells': N, 'n_triples': len(trips),
        'key_layers': list(KEY_LAYERS),
        'behaviour_gate': {q: '%d/%d' % tuple(beh[q]) for q in QUERIES},
        'median': {k: round(v, 4) for k, v in med.items()},
        'iqr': {k: [round(q1[k], 4), round(q3[k], 4)] for k in med},
        'h3_hist_10bins': hist, 'h3_peaks': int(pk),
        'profile': prof,
        'dual_arm': {'n_pairs': len(dual), 'delta_h3_median': round(a5_med, 4),
                     'delta_h3_iqr': [round(float(np.percentile(d_h3, 25)), 4),
                                      round(float(np.percentile(d_h3, 75)), 4)]},
        'head_split': {'head_acc_top8': {int(h): round(float(head_acc[h]), 4) for h in top},
                       'head_acc_median': round(float(np.median(head_acc)), 4),
                       'n_heads_above_080': int((head_acc > 0.80).sum()),
                       'n_heads_below_050': int((head_acc < 0.50).sum()),
                       'unanim_signal_only_median': round(float(np.median(uni_sig)), 4)
                       if len(uni_sig) else None,
                       'n_signal_obs': int(len(uni_sig))},
        'k_curve': {int(li): {int(k): list(v) for k, v in d.items()} for li, d in kcurve.items()},
        'head_acc_per_layer': {int(li): [round(float(x), 3) for x in head_acc_L[li]]
                               for li in KEY_LAYERS},
        'sink_diag': {'L28_sink_median': round(float(np.median(sk)), 4),
                      'within_cell_head_sd': round(sd_head, 4),
                      'per_head_sink_median': round(float(np.median(sk.mean(0))), 4),
                      'per_head_sink_range': [round(float(sk.mean(0).min()), 4),
                                              round(float(sk.mean(0).max()), 4)],
                      'corr_sink_vs_headacc_L28': round(float(np.corrcoef(sk.mean(0), h28)[0, 1]), 4)},
        'position_diag': {'pos0_median': round(float(np.median(sink0)), 4),
                          'pos1_median': round(float(np.median(sink1)), 4),
                          'cover_plus_sinks': round(med['cover'] + float(np.median(sink0 + sink1)), 4)},
        'criteria': {'A1': A1, 'A2': A2, 'A3': A3, 'A4': A4, 'A5': A5},
        'verdict': verdict,
        'note': '形状判据：三槽熵/单点质量/头一致性；冲突臂用等长非查询对照防长度混杂；'
                'unanim 是“多维度”假设的直接检验（Q4-R 只看过跨头求和后的 argmax）。',
        'elapsed_s': round(time.time() - t0, 1),
    }
    provenance.write_result('pat_q5a', rec, model_path=MODEL_PATH, seed=SEED, tool='pat_q5a.py')
    print('完成 %.0fs' % (time.time() - t0))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
