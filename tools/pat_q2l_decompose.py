# -*- coding: utf-8 -*-
"""Q2-L  Δ 的成分分解：末位读出的锥是「角色锥」还是「实体身份锥」

问题（Q2-C2c 遗留，预注册于 PAT_Q2C_FINDINGS_20261101.md 末节）
------------------------------------------------------------------
上一轮测到 L33 末位读出有方向选择性，有效方向是 cos≈0.60（≈53°）为边界的
「角度门锥」，且锥逐样本朝向不同（cos(Δ_i,Δ_j) 均值 0.245）。但
Δ = h(clean) − h(corrupt) 由「实体互换」构造，同时混了三样东西：

  (1) 答案词的**词法身份**（张三 → 李四）
  (2) **角色归属**（谁在施事位）
  (3) 大量与答案无关的**句子特异内容**

三者从未分开。若 (1) 就足以解释全部效应，则「锥」只是反嵌入矩阵的几何，
不是角色机制，Q2-C2c 第七节推论必须改写。

设计：全部同基座（本轮修正的关键）
---------------------------------
第一版设计里，实体差分在 S1→S3 上测量却注入 S2，属于**跨上下文移植**；
而角色差分在 S2→S4 上测量并注入 S2，属于**原位移植**。两者差异可能只是
移植是否匹配，读成「实体 vs 角色」会错。故全部改为同一基座 B：

  B  = (Y, X, T)      答案 Y（corrupt 句）
  P1 = (X, Y, T)      答案 X  —— 受事也换（角色 + 身份同时变）
  P2 = (Y', X, T)     答案 Y' —— 受事不变，只换施事身份；Y' 不在句中
  P3 = (Y, X, T')     答案 Y  —— 只换宾语，答案不变（特异性对照）

  且 B 的受事位置放的是 X，故 P1 的目标 X **在上下文内**；P2 的目标 Y'
  **不在上下文内**。这一对因此同时回答「是否需要目标词已出现」。

全部 Δ 都从 B 测得、都注入 B 的同一个末位，无移植错配。

方向清单（均在 L33 末位注入，位移长度统一为 p·‖d1‖，符合 D7）
------------------------------------------------------------
  d1     = h(P1) − h(B)                     完整角色+身份差分（基线）
  d2     = h(P2) − h(B)                     仅身份差分，目标不在句中
  d3     = h(P3) − h(B)                     内容变、答案不变（应无扰动）
  lex1   = unit(E[X]  − E[Y])               纯词法方向，目标 X
  lex2   = unit(E[Y'] − E[Y])               纯词法方向，目标 Y'
  orth1  = unit(d1 − (d1·û_lex1) û_lex1)    d1 去掉词法分量后的残差，目标 X
  orth2  = unit(d2 − (d2·û_lex2) û_lex2)    d2 去掉词法分量后的残差，目标 Y'
  rand   = 随机单位向量                     同长度对照，目标 X

  orth1 是本轮最关键的一条：若它单独翻转，说明除词法外还存在独立可读方向；
  若它失效而 lex1 有效，则上一轮的「锥」就是词法锥。

判据（预注册，三块独立报告）
----------------------------
门 E0：d1@p=1.0 翻转率 < 0.80 ⇒ 整轮作废（管线或行为失效）。
门 S0：d3@p=1.0 的「答案不变率」 < 0.80 ⇒ 末位 patch 非特异，判据不可用。

块一 夹角（描述性）
  cA = cos(d1, d2) 中位；cL1 = cos(d1, lex1)；cL2 = cos(d2, lex2)
  记录 min/median/max/IQR；极差 >=0.30 时主判据改用中位（D5）。

块二 词法分解（判决）
  O1 全词法：flip(orth1) <= 0.30 且 flip(lex1) >= 0.80
  O2 有结构分量：flip(orth1) >= 0.70
  O3 中间：其余

块三 目标实体是否须在上下文
  C1 须在上下文：flip(d2) <= 0.30 且 flip(d1) >= 0.70
  C2 不须：flip(d2) >= 0.70
  C3 中间：其余

成本：48 样本 × (4 采集 + 8 方向 × 3 p) ≈ 1350 次前向。

用法
----
  SMOKE=1 .venv/Scripts/python.exe -X utf8 tools/pat_q2l_decompose.py
  .venv/Scripts/python.exe -X utf8 tools/pat_q2l_decompose.py
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
from pat_q2c1_patch import build_rec, patch_layer
from pat_q1_unbind import OBJECTS, PERSONS

SEED = 20261114
SMOKE = os.environ.get('SMOKE', '') == '1'
N_TRIP = 4 if SMOKE else 24
FORMS = ('A', 'C')
READ_LAYER = 33
PS = (0.5, 1.0) if SMOKE else (0.5, 0.75, 1.0)
N_BOOT = 5000
COS_SAME = 0.60
MODEL_PATH = 'D:/develop/model/Qwen/Qwen3-4B'

GROUPS = ('d1', 'd2', 'd3', 'lex1', 'lex2', 'orth1', 'orth2', 'rand')
SUBS = (0.0, 0.25, 0.5, 0.75, 1.0)


def boot_mean(vals, rng):
    a = np.asarray(vals, dtype=float)
    if a.size == 0:
        return float('nan'), float('nan'), float('nan')
    idx = rng.integers(0, a.size, size=(N_BOOT, a.size))
    m = a[idx].mean(axis=1)
    return float(a.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def dist(vals):
    a = np.asarray(vals, dtype=float)
    if a.size == 0:
        return {'n': 0}
    return {'n': int(a.size), 'mean': round(float(a.mean()), 4),
            'median': round(float(np.median(a)), 4),
            'min': round(float(a.min()), 4), 'max': round(float(a.max()), 4),
            'iqr': round(float(np.percentile(a, 75) - np.percentile(a, 25)), 4),
            'range': round(float(a.max() - a.min()), 4)}


def build_triples(n: int):
    """自建三元组，覆盖全部 (i, i+3) 人物对。

    D8：triples(24) 实际只含 4 对人物（i∈0..3），实体身份与配对索引强耦合，
    分解研究必须先把实体多样性拉满。此处 12 对 × 2 宾语 = 24。
    """
    pairs = [(PERSONS[i], PERSONS[(i + 3) % len(PERSONS)]) for i in range(len(PERSONS))]
    out, rnd = [], 0
    while len(out) < n:
        for idx, (a, b) in enumerate(pairs):
            if len(out) >= n:
                break
            out.append((a, b, OBJECTS[(idx + rnd * 3) % len(OBJECTS)]))
        rnd += 1
    return out


def pick_alt_obj(tok, obj, exclude):
    tgt = len(tok(obj, add_special_tokens=False)['input_ids'])
    for cand in OBJECTS:
        if cand == exclude:
            continue
        if len(tok(cand, add_special_tokens=False)['input_ids']) == tgt:
            return cand
    return None


def pick_new_person(tok, used):
    """同 token 数、首 token 不同、不在 used 里的人名。"""
    base = tok(used[0], add_special_tokens=False)['input_ids']
    b0 = base[0]
    for cand in PERSONS:
        if cand in used:
            continue
        ids = tok(cand, add_special_tokens=False)['input_ids']
        if len(ids) == len(base) and ids[0] != b0:
            return cand
    return None


@torch.no_grad()
def collect_last(model, dev, ids):
    o = model(input_ids=torch.tensor([ids], device=dev), use_cache=False,
              output_hidden_states=True)
    h = o.hidden_states[READ_LAYER + 1][0, len(ids) - 1, :].clone()
    lp = torch.log_softmax(o.logits[0, -1].float(), dim=-1)
    return h, lp


@torch.no_grad()
def predict(model, dev, ids, pos, vec, layer):
    with patch_layer(layer, pos, vec):
        o = model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
    return int(o.logits[0, -1].argmax())


def unitize(v):
    n = float(v.norm())
    return v / n if n > 1e-9 else v * 0.0


def main() -> int:
    from server import live_analysis_service as las

    t0 = time.time()
    las._load_async()
    while las._state['status'] in ('idle', 'loading'):
        if time.time() - t0 > 900:
            raise RuntimeError('模型加载超时')
        time.sleep(2.0)
    model, tok = las._state['model'], las._state['tokenizer']
    dev = next(model.parameters()).device
    layers = model.model.layers
    emb = model.get_input_embeddings().weight

    print('=== Q2-L  Δ 成分分解（全部同基座）===')
    print('freeze: READ_LAYER=%d  p=%s  n_trip=%d  位移长度统一 = p*||d1||'
          % (READ_LAYER, PS, N_TRIP))

    trips = build_triples(N_TRIP)
    print('人物对覆盖: %d 对 / %d 条' % (len({(t[0], t[1]) for t in trips}), len(trips)))

    recs, gate = [], {'n': 0, 'b_ok': 0, 'p1_ok': 0, 'p2_ok': 0, 'p3_ok': 0}
    skipped = {'new': 0, 'obj': 0, 'len': 0}

    for form in FORMS:
        for trip in trips:
            X, Y, T = trip
            Yp = pick_new_person(tok, [Y, X])
            Tp = pick_alt_obj(tok, T, T)
            if Yp is None:
                skipped['new'] += 1
                continue
            if Tp is None:
                skipped['obj'] += 1
                continue

            B = build_rec(tok, (Y, X, T), form)
            P1 = build_rec(tok, (X, Y, T), form)
            P2 = build_rec(tok, (Yp, X, T), form)
            P3 = build_rec(tok, (Y, X, Tp), form)
            n = len(B['ids'])
            if any(len(s['ids']) != n for s in (P1, P2, P3)):
                skipped['len'] += 1
                continue

            tY = tok(Y, add_special_tokens=False)['input_ids'][0]
            tX = tok(X, add_special_tokens=False)['input_ids'][0]
            tYp = tok(Yp, add_special_tokens=False)['input_ids'][0]

            hB, lpB = collect_last(model, dev, B['ids'])
            h1, lp1 = collect_last(model, dev, P1['ids'])
            h2, lp2 = collect_last(model, dev, P2['ids'])
            h3, lp3 = collect_last(model, dev, P3['ids'])

            gate['n'] += 1
            gate['b_ok'] += int(float(lpB[tY]) > max(float(lpB[tX]), float(lpB[tYp])))
            gate['p1_ok'] += int(float(lp1[tX]) > max(float(lp1[tY]), float(lp1[tYp])))
            gate['p2_ok'] += int(float(lp2[tYp]) > max(float(lp2[tY]), float(lp2[tX])))
            gate['p3_ok'] += int(float(lp3[tY]) > max(float(lp3[tX]), float(lp3[tYp])))

            recs.append({'form': form, 'trip': trip, 'Yp': Yp, 'Tp': Tp,
                         'ids': B['ids'], 'last': n - 1, 'dt': hB.dtype,
                         'hB': hB.float(), 'h1': h1.float(), 'h2': h2.float(),
                         'h3': h3.float(), 'tX': tX, 'tY': tY, 'tYp': tYp})

    print('行为门：B->Y %d/%d  P1->X %d/%d  P2->Yp %d/%d  P3->Y %d/%d   跳过 %s'
          % (gate['b_ok'], gate['n'], gate['p1_ok'], gate['n'],
             gate['p2_ok'], gate['n'], gate['p3_ok'], gate['n'], skipped))
    if not recs:
        print('无可分析样本，退出')
        return 1

    # ---------- 方向构造 ----------
    rng_dir = np.random.default_rng(SEED + 1)
    keep = []
    for r in recs:
        d1 = r['h1'] - r['hB']
        d2 = r['h2'] - r['hB']
        d3 = r['h3'] - r['hB']
        if float(d1.norm()) < 1e-6 or float(d2.norm()) < 1e-6:
            continue
        lex1 = unitize((emb[r['tX']] - emb[r['tY']]).float().detach())
        lex2 = unitize((emb[r['tYp']] - emb[r['tY']]).float().detach())
        orth1 = unitize(d1 - float(torch.dot(d1, lex1)) * lex1)
        orth2 = unitize(d2 - float(torch.dot(d2, lex2)) * lex2)
        vz = rng_dir.standard_normal(d1.shape[0]).astype(np.float32)
        rnd = unitize(torch.from_numpy(vz)).to(d1.device)
        r.update({'d1': d1, 'd2': d2, 'd3': d3, 'lex1': lex1, 'lex2': lex2,
                  'orth1': orth1, 'orth2': orth2, 'rand': rnd,
                  'n1': float(d1.norm())})
        keep.append(r)
    recs = keep

    cos_d1d2 = [float(torch.dot(unitize(r['d1']), unitize(r['d2']))) for r in recs]
    cos_d1l1 = [float(torch.dot(unitize(r['d1']), r['lex1'])) for r in recs]
    cos_d2l2 = [float(torch.dot(unitize(r['d2']), r['lex2'])) for r in recs]
    cos_l1l2 = [float(torch.dot(r['lex1'], r['lex2'])) for r in recs]
    nrm_ratio = [float(r['d2'].norm()) / r['n1'] for r in recs]

    print('\n=== 块一 夹角（D5：先看分布，均值仅参考）===')
    for nm, arr in (('cos(d1,d2)', cos_d1d2), ('cos(d1,lex1)', cos_d1l1),
                    ('cos(d2,lex2)', cos_d2l2), ('cos(lex1,lex2)', cos_l1l2)):
        d = dist(arr)
        print('  %-16s n=%-3d mean=%+.3f median=%+.3f min=%+.3f max=%+.3f IQR=%.3f'
              % (nm, d['n'], d['mean'], d['median'], d['min'], d['max'], d['iqr']))
    print('  ||d2||/||d1|| 中位 %.3f ｜ ||d3||/||d1|| 中位 %.3f ｜ ||d1|| 中位 %.2f'
          % (float(np.median(nrm_ratio)),
             float(np.median([float(r['d3'].norm()) / r['n1'] for r in recs])),
             float(np.median([r['n1'] for r in recs]))))

    by_key = {}
    for r in recs:
        by_key.setdefault(r['trip'], {})[r['form']] = r
    ratio_voice = []
    for trip, dd in by_key.items():
        if 'A' in dd and 'C' in dd:
            dv = dd['A']['hB'] - dd['C']['hB']
            ratio_voice.append(float(dv.norm()) / max(dd['A']['n1'], 1e-9))
    if ratio_voice:
        d = dist(ratio_voice)
        print('  ||D_voice||/||d1||（同答案、不同语序）n=%d 中位 %.3f' % (d['n'], d['median']))

    # ---------- 干预 ----------
    rng = np.random.default_rng(SEED + 2)
    hits = {g: {p: {} for p in PS} for g in GROUPS}
    nat = {g: {} for g in ('d1', 'd2', 'd3')}     # 自然长度移植（原 Δ 原样注入）
    subs = {lam: {} for lam in SUBS}              # 逐级剥离词法分量，固定长度
    base_re = {'b': 0, 'n': 0}
    pred_hist = {'d3': {}, 'rand': {}}

    TGT = {'d1': 'tX', 'd2': 'tYp', 'd3': 'tY', 'lex1': 'tX', 'lex2': 'tYp',
           'orth1': 'tX', 'orth2': 'tYp', 'rand': 'tX'}

    for k, r in enumerate(recs):
        pos, ids, layer, dt = r['last'], r['ids'], layers[READ_LAYER], r['dt']
        base_re['n'] += 1
        base_re['b'] += int(predict(model, dev, ids, pos, r['hB'].to(dt), layer) == r['tY'])

        for p in PS:
            L = p * r['n1']
            for g in GROUPS:
                v = (r['hB'] + L * r[g]).to(dt)
                pred = predict(model, dev, ids, pos, v, layer)
                hits[g][p][k] = int(pred == r[TGT[g]])
                if p == PS[-1] and g in pred_hist:
                    pred_hist[g][pred] = pred_hist[g].get(pred, 0) + 1

        # 自然长度移植：把测得的 Δ 原样注入，不做长度归一
        for g in ('d1', 'd2', 'd3'):
            v = (r['hB'] + r[g]).to(dt)
            nat[g][k] = int(predict(model, dev, ids, pos, v, layer) == r[TGT[g]])

        # 逐级剥离词法分量（固定长度 = ||d1||）
        u1 = unitize(r['d1'])
        proj = float(torch.dot(r['d1'], r['lex1']))
        for lam in SUBS:
            dsub = r['d1'] - lam * proj * r['lex1']
            v = (r['hB'] + r['n1'] * unitize(dsub)).to(dt)
            subs[lam][k] = int(predict(model, dev, ids, pos, v, layer) == r['tX'])

    rng_b = np.random.default_rng(SEED + 20)
    print('\n=== 块二/三 干预：同长度（p*||d1||）下的命中率 ===')
    print('  （d3 的目标是「答案保持 Y」，即不变率）')
    print('   p    ' + ''.join('%-7s' % g for g in GROUPS))
    curves = {}
    for p in PS:
        row = []
        for g in GROUPS:
            m_, lo, hi = boot_mean(list(hits[g][p].values()), rng_b)
            curves[(g, p)] = (m_, lo, hi)
            row.append('%.3f' % m_)
        print('  %.2f  ' % p + ''.join('%-7s' % v for v in row))
    print('  注入原值复核：B->Y %d/%d' % (base_re['b'], base_re['n']))

    print('\n=== 自然长度移植（把测得的 Δ 原样注入，不做归一）===')
    nat_m = {}
    for g in ('d1', 'd2', 'd3'):
        m_, lo, hi = boot_mean(list(nat[g].values()), rng_b)
        nat_m[g] = m_
        print('  %-4s 命中 %.3f  CI[%.3f,%.3f]  (d3 的命中=答案保持不变)' % (g, m_, lo, hi))

    print('\n=== 逐级剥离词法分量（固定长度 ||d1||，p=1.0）===')
    sub_m = {}
    for lam in SUBS:
        m_, lo, hi = boot_mean(list(subs[lam].values()), rng_b)
        sub_m[lam] = m_
        print('  λ=%.2f  翻转 %.3f  CI[%.3f,%.3f]' % (lam, m_, lo, hi))

    print('\n=== 未命中时模型输出了什么（p=%.2f）===' % PS[-1])
    for g in ('d3', 'rand'):
        top = sorted(pred_hist[g].items(), key=lambda kv: -kv[1])[:3]
        print('  %-4s -> %s' % (g, [(tok.decode([i]).strip()[:4], c) for i, c in top]))

    def paired(a, b):
        keys = sorted(set(a) & set(b))
        dd = np.array([a[i] - b[i] for i in keys], dtype=float)
        idx = rng_b.integers(0, dd.size, size=(N_BOOT, dd.size))
        m = dd[idx].mean(axis=1)
        return float(dd.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

    print('\n=== 配对差（对照 d1）===')
    detail = []
    for p in PS:
        row = {'p': p}
        for g in ('d2', 'lex1', 'orth1', 'lex2', 'orth2', 'rand'):
            dd = paired(hits[g][p], hits['d1'][p])
            row[g] = [round(dd[0], 4), round(dd[1], 4), round(dd[2], 4)]
        detail.append(row)
        print('  p=%.2f  ' % p + ' '.join('%s%+.2f[%+.2f,%+.2f]'
              % (g, row[g][0], row[g][1], row[g][2]) for g in row if g != 'p'))

    # ---------- 判据 ----------
    P = PS[-1]
    f_d1 = curves[('d1', P)][0]
    f_d2 = curves[('d2', P)][0]
    f_lex1 = curves[('lex1', P)][0]
    f_orth1 = curves[('orth1', P)][0]
    stay_d3 = nat_m['d3']              # 特异性改用「自然长度」口径
    e0 = f_d1 >= 0.80 and nat_m['d1'] >= 0.80
    s0 = stay_d3 >= 0.80

    if f_orth1 <= 0.30 and f_lex1 >= 0.80:
        o_code, o_desc = 'O1', '全词法：去词法分量后失效，锥=反嵌入几何'
    elif f_orth1 >= 0.70:
        o_code, o_desc = 'O2', '存在独立结构分量：去词法后仍 >=0.70'
    else:
        o_code, o_desc = 'O3', '中间：词法主、结构残留'

    if f_d2 <= 0.30 and f_d1 >= 0.70:
        c_code, c_desc = 'C1', '目标须已在上下文内'
    elif f_d2 >= 0.70:
        c_code, c_desc = 'C2', '目标不必在上下文内'
    else:
        c_code, c_desc = 'C3', '中间'

    # 块四：剥离曲线形状（判决词法是否为唯一承载）
    s0_, s1_ = sub_m[0.0], sub_m[1.0]
    if s1_ <= 0.30 and s0_ >= 0.80:
        d_code, d_desc = 'D1', '一剥即崩（阶跃）：词法分量是必要条件'
    elif s1_ >= 0.70:
        d_code, d_desc = 'D2', '剥离后仍有效：词法非必要，存在等价通路'
    else:
        d_code, d_desc = 'D3', '中间：部分依赖'

    cA = float(np.median(cos_d1d2))
    cL1 = float(np.median(cos_d1l1))
    cL2 = float(np.median(cos_d2l2))

    print('\n=== 判据 ===')
    print('  门 E0（匹配 p=%.2f 命中 %.3f，自然长度 %.3f，均需 >=0.80）: %s'
          % (P, f_d1, nat_m['d1'], 'PASS' if e0 else 'FAIL 整轮作废'))
    print('  门 S0（d3 自然长度下答案不变 %.3f >=0.80）: %s'
          % (stay_d3, 'PASS' if s0 else 'FAIL 末位 patch 非特异'))
    print('  块二 %s %s   [orth1=%.3f lex1=%.3f d1=%.3f]'
          % (o_code, o_desc, f_orth1, f_lex1, f_d1))
    print('  块三 %s %s   [d2=%.3f d1=%.3f]' % (c_code, c_desc, f_d2, f_d1))
    print('  块四 %s %s   [λ=0 %.3f → λ=1 %.3f]' % (d_code, d_desc, s0_, s1_))
    print('  夹角 median: cos(d1,d2)=%+.3f  cos(d1,lex1)=%+.3f  cos(d2,lex2)=%+.3f'
          % (cA, cL1, cL2))

    verdict = '%s %s（orth1=%.3f lex1=%.3f）｜%s %s（d2=%.3f）｜%s %s｜E0 %s S0 %s' % (
        o_code, o_desc, f_orth1, f_lex1, c_code, c_desc, f_d2, d_code, d_desc,
        'PASS' if e0 else 'FAIL', 'PASS' if s0 else 'FAIL')
    print('\nverdict: %s' % verdict)

    payload = {
        'read_layer': READ_LAYER, 'ps': list(PS), 'n_trip': N_TRIP,
        'n_samples': len(recs), 'forms': list(FORMS),
        'gate': gate, 'skipped': skipped, 'base_reinject': base_re,
        'cos': {'d1_d2': dist(cos_d1d2), 'd1_lex1': dist(cos_d1l1),
                'd2_lex2': dist(cos_d2l2), 'lex1_lex2': dist(cos_l1l2)},
        'norm': {'d1_median': round(float(np.median([r['n1'] for r in recs])), 4),
                 'd2_over_d1': dist(nrm_ratio),
                 'voice_ratio': dist(ratio_voice) if ratio_voice else {}},
        'curves': {'%s@p%.2f' % (g, p): [round(curves[(g, p)][0], 4),
                                          round(curves[(g, p)][1], 4),
                                          round(curves[(g, p)][2], 4)]
                   for g in GROUPS for p in PS},
        'natural': {g: round(nat_m[g], 4) for g in nat_m},
        'sub_lex': {'%.2f' % lam: round(sub_m[lam], 4) for lam in SUBS},
        'pred_hist': {g: {str(i): c for i, c in pred_hist[g].items()} for g in pred_hist},
        'paired': detail,
        'block2': {'code': o_code, 'desc': o_desc, 'orth1': round(f_orth1, 4),
                   'lex1': round(f_lex1, 4), 'd1': round(f_d1, 4)},
        'block3': {'code': c_code, 'desc': c_desc, 'd2': round(f_d2, 4)},
        'block4': {'code': d_code, 'desc': d_desc, 'lam0': round(s0_, 4),
                   'lam1': round(s1_, 4)},
        'gate_E0': bool(e0), 'spec_S0': bool(s0),
        'verdict': verdict,
    }
    path = provenance.write_result('pat_q2l_decompose', payload,
                                   model_path=MODEL_PATH, seed=SEED,
                                   tool='tools/pat_q2l_decompose.py')
    print('落盘 %s' % path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
