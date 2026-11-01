#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Q2-C2: 末位读出的方向结构判决 —— 答案身份是线性方向，还是位置指针？

背景
----
Q2-C1（res 93daf69b / ledger #450）在 L33 定位到读出点：
  - L0–L29 patch agent 位置 -> 翻转 1.00；L33 起精确 0.00（输入通道关闭）
  - L0–L25 patch 末位 -> 0.00；L33 起精确 1.00（输出由末位决定）
  - 同段 L28–L35 消融全部 133 头 -> 增量 0.000（Q2-D2）
结论：容量分布式、读出单点。本轮问的是读出端的表示形式。

设计（在 corrupt 句上，只改 layers[33] 输出的末位向量）
--------------------------------------------------
记 h_c = clean 末位激活、h_x = corrupt 末位激活、D_i = h_c - h_x、
    n_i = ||D_i||、u_i = D_i / n_i。

  A 本句差分方向      vec = h_x + a * D_i                    （a 以 n_i 为单位）
  S 共享方向          vec = h_x + a * n_i * m_hat_i            （m_hat 为**留一**估计：
                                                                 用其他全部样本的 D 求均值方向）
  R 特异残差          vec = h_x + a * resid_i                  （resid = D_i 去掉其在 m_hat 上的投影）
  B 随机方向（幅度匹配）vec = h_x + a * n_i * u_rand           （u_rand 每样本固定）
  D 离线方向相似度    cos(D_i, D_j) 分布 —— 决定 S/R 分解是否必要

设计修订记录（SMOKE 阶段，执行前，D6）
----------------------------------
初版第三组为「C 跨三元组迁移」。SMOKE 实测 cos(D_i,D_j) 均值 0.936（0.84–0.99），
方向几乎共线，迁移成功是平凡的、无区分力。故改为 S/R 分解：
共线本身就说明「存在共享分量」，真正的问题是**翻转由共享分量还是特异分量驱动**。
m_hat 必须留一估计（用其他样本），否则 S 组含自身分量、退化为 A 的子集。

预注册判据（三块独立，各自必报；不允许合并成单一 verdict）
----------------------------------------------------
块一 形状（只看 A 随 a 的曲线）
  F0 执行失效：A(a=1.0) < 0.90 —— a=1 等价于恒等还原（Q2-C1 已验证 1.00），
     未达标说明 hook 或采集错，整轮作废、不作任何机制解读。
  F1 早期饱和：A(最小 a) >= 0.90 —— 末位贴近决策边界，细微推动即翻。
  F2 线性可加：A 单调不减，A(最小 a) <= 0.60，A(1.0) >= 0.90。
  F3 阈值型：A(a<=0.75) 全部 <= 0.30 而 A(1.0) >= 0.90 —— 需完整抵消，
     无中间态；末位表示不是可加线性量。
  F4 中间型：以上皆不满足（明确记录，不许硬塞）。

块二 方向特异性（A 与 B 在同 a 下的配对差）
  G1 方向特异：max_a (A-B) >= 0.30 且 CI 下界 > 0
  G2 仅幅度效应：所有 a 的 (A-B) <= 0.20
  G3 弱特异：0.20 < max_a (A-B) < 0.30

块三 分解（S 共享方向 vs R 特异残差，均取 a=1.0，同幅度）
  H1 共享驱动：S(1.0) >= 0.70 且 R(1.0) <= 0.30
     => 「答案是谁」在此层被压成跨实体共享的单一角色方向，实体身份不承载翻转。
        这是**方向假说在读出端的强形式**：角色 = 一个方向，与实体无关。
  H2 特异驱动：S(1.0) <= 0.30 且 R(1.0) >= 0.70
     => 翻转由每样本的特异分量驱动，共享分量不承载（尽管方向高度共线）。
  H3 双分量协同：S、R 均 >= 0.70，或均 <= 0.30 而 A(1.0) >= 0.90
     => 翻转需两分量同时到位，末位读出对分解不是线性的。
        后者尤其重要：任一分量单独都不够、合起来才够 ⇒ 非线性交互。
  H4 中间：以上皆不满足（明确记录）。

预注册预测（执行前写死，输了也记账）
--------------------------------
预测 F2 + G1 + H1：末位身份是线性可加、方向特异，且**由跨实体共享的单一方向驱动**。
若 H1 成立，则「角色」在本层就是一个可命名的方向，图谱形态回到方向表/特征清单；
若 F3 成立（阈值），则末位读出非线性，方向假说在读出端不成立，须改非线性干预；
若 H3-后半（both <=0.30 而 A 高）成立，则说明两分量必须同时到位，
这将是本线第一次测到**真正的非线性交互**。

成本
----
48 对（24 三元组 x 2 句式）x (2 采集 + 4A + 4B + 1C) ~= 530 次前向。
"""
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / 'tools')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import provenance
from pat_q2c1_patch import build_rec, patch_layer, pos_map
from pat_q1_unbind import triples

SEED = 20261111
SMOKE = os.environ.get('SMOKE', '') == '1'
N_TRIP = 4 if SMOKE else 24
FORMS = ('A', 'C')
ALPHAS = (0.5, 0.75, 1.0) if SMOKE else (0.25, 0.5, 0.75, 1.0)
READ_LAYER = 33               # Q2-C1 定位的读出点（layers[33] 的输出）
N_BOOT = 5000
MODEL_PATH = 'D:/develop/model/Qwen/Qwen3-4B'


def boot_mean(vals, rng):
    a = np.asarray(vals, dtype=float)
    if a.size == 0:
        return float('nan'), float('nan'), float('nan')
    idx = rng.integers(0, a.size, size=(N_BOOT, a.size))
    m = a[idx].mean(axis=1)
    return float(a.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_diff(a, b, rng):
    """配对差 a-b 的均值与 95% CI。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    if d.size == 0:
        return float('nan'), float('nan'), float('nan')
    idx = rng.integers(0, d.size, size=(N_BOOT, d.size))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


@torch.no_grad()
def forward_collect(model, dev, ids):
    """返回 (末位激活[该层输出]、log_softmax)。"""
    o = model(input_ids=torch.tensor([ids], device=dev), use_cache=False,
              output_hidden_states=True)
    h = o.hidden_states[READ_LAYER + 1][0, len(ids) - 1, :].clone()
    lp = torch.log_softmax(o.logits[0, -1].float(), dim=-1)
    return h, lp


@torch.no_grad()
def forward_argmax(model, dev, ids):
    o = model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
    return int(o.logits[0, -1].argmax())


@torch.no_grad()
def patch_predict(model, dev, ids, pos, vec, layer):
    with patch_layer(layer, pos, vec):
        o = model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
    return int(o.logits[0, -1].argmax())


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

    print('=== Q2-C2 末位方向结构判决 ===')
    print('freeze: READ_LAYER=%d (layers[%d] 输出)  alpha=%s  n_trip=%d  forms=%s'
          % (READ_LAYER, READ_LAYER, ALPHAS, N_TRIP, '/'.join(FORMS)))
    print('判据：块一形状 F0-F4 / 块二方向特异 G1-G3 / 块三迁移 H1-H3')

    trips = triples(N_TRIP)
    # ---------- Phase 1: 采集 ----------
    recs = []
    gate = {'clean_ok': 0, 'corrupt_ok': 0, 'n': 0, 'skip_len': 0, 'skip_norm': 0}
    for form in FORMS:
        for ti, trip in enumerate(trips):
            clean = build_rec(tok, trip, form)
            corrupt = build_rec(tok, (trip[1], trip[0], trip[2]), form)
            if len(clean['ids']) != len(corrupt['ids']):
                gate['skip_len'] += 1
                continue
            pm = pos_map(tok, clean)
            last = pm['last']
            ta = tok(trip[0], add_special_tokens=False)['input_ids'][0]
            tb = tok(trip[1], add_special_tokens=False)['input_ids'][0]

            hc, lp_c = forward_collect(model, dev, clean['ids'])
            hx, lp_x = forward_collect(model, dev, corrupt['ids'])
            gate['clean_ok'] += int(float(lp_c[ta]) > float(lp_c[tb]))
            gate['corrupt_ok'] += int(float(lp_x[tb]) > float(lp_x[ta]))
            gate['n'] += 1

            dt = hc.dtype
            d = (hc.float() - hx.float())
            nrm = float(d.norm())
            if nrm < 1e-6:
                gate['skip_norm'] += 1
                continue
            recs.append({'form': form, 'ti': ti, 'trip': trip, 'ids': corrupt['ids'],
                         'last': last, 'ta': ta, 'tb': tb, 'dt': dt,
                         'hx': hx.float(), 'delta': d, 'nrm': nrm})

    print('行为门：clean %d/%d 答 agent；corrupt %d/%d 答 recipient；长度不等跳过 %d；差分为零 %d'
          % (gate['clean_ok'], gate['n'], gate['corrupt_ok'], gate['n'],
             gate['skip_len'], gate['skip_norm']))
    print('可用配对 n=%d（每格样本数）' % len(recs))
    if not recs:
        raise RuntimeError('无可用配对，终止')

    # ---------- Phase 2: D 组离线方向相似度（先算，决定 C 的解释力）----------
    cos_pairs = []
    for form in FORMS:
        sub = [r for r in recs if r['form'] == form]
        for i in range(len(sub)):
            for j in range(i + 1, len(sub)):
                a, b = sub[i]['delta'], sub[j]['delta']
                c = float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-9))
                cos_pairs.append(c)
    cos_arr = np.asarray(cos_pairs, dtype=float)
    cos_mean = float(cos_arr.mean()) if cos_arr.size else float('nan')
    cos_p50 = float(np.median(cos_arr)) if cos_arr.size else float('nan')
    cos_max = float(cos_arr.max()) if cos_arr.size else float('nan')
    print('\n=== D 组：跨三元组差分方向余弦（离线，%d 对）===' % cos_arr.size)
    print('  mean=%.3f  median=%.3f  max=%.3f  min=%.3f'
          % (cos_mean, cos_p50, cos_max, cos_arr.min() if cos_arr.size else float('nan')))
    print('  解释约束：cos 均值 >= 0.70 时，H1 不可作为「共享方向轴」的证据')

    # ---------- Phase 3: A/B/C 干预 ----------
    rng = np.random.default_rng(SEED)
    # 随机方向流与配对流分离，保证可复现
    rng_dir = np.random.default_rng(SEED + 1)

    rand_unit = {}
    for k, r in enumerate(recs):
        v = rng_dir.standard_normal(r['delta'].shape[0]).astype(np.float32)
        n = float(np.linalg.norm(v))
        rand_unit[k] = torch.from_numpy(v / max(n, 1e-9)).to(r['hx'].device)

    # 留一共享方向 m_hat_i：用「其他全部样本」的差分均值方向（不含自身）
    all_delta = torch.stack([r['delta'] for r in recs])
    m_hat = {}
    for k in range(len(recs)):
        others = torch.cat([all_delta[:k], all_delta[k + 1:]], dim=0).mean(dim=0)
        n = float(others.norm())
        m_hat[k] = others / max(n, 1e-9)
    # 诊断：自身差分与该留一方向的夹角余弦
    self_cos = [float(torch.dot(recs[k]['delta'], m_hat[k])
                      / (recs[k]['nrm'] + 1e-9)) for k in range(len(recs))]
    print('\n留一共享方向诊断：cos(D_i, m_hat_-i) mean=%.3f min=%.3f'
          % (float(np.mean(self_cos)), float(np.min(self_cos))))

    res_A = {a: [] for a in ALPHAS}
    res_B = {a: [] for a in ALPHAS}
    res_S = {a: [] for a in ALPHAS}
    res_R = {a: [] for a in ALPHAS}
    resid_ratio = []
    for k, r in enumerate(recs):
        hx, d, nrm = r['hx'], r['delta'], r['nrm']
        pos, ids, ta, dt, layer = r['last'], r['ids'], r['ta'], r['dt'], layers[READ_LAYER]
        m = m_hat[k]
        proj = float(torch.dot(d, m))
        resid = d - proj * m
        resid_ratio.append(float(resid.norm()) / max(nrm, 1e-9))

        # 行为门复核：未 patch 时应答 recipient
        base_pred = forward_argmax(model, dev, ids)
        if base_pred == ta:
            print('  !! 基线已答 agent，跳过 k=%d' % k)
            continue

        for a in ALPHAS:
            vA = (hx + a * d).to(dt)
            res_A[a].append(int(patch_predict(model, dev, ids, pos, vA, layer) == ta))
            vB = (hx + a * nrm * rand_unit[k]).to(dt)
            res_B[a].append(int(patch_predict(model, dev, ids, pos, vB, layer) == ta))
            vS = (hx + a * nrm * m).to(dt)
            res_S[a].append(int(patch_predict(model, dev, ids, pos, vS, layer) == ta))
            vR = (hx + a * resid).to(dt)
            res_R[a].append(int(patch_predict(model, dev, ids, pos, vR, layer) == ta))

    print('特异残差占比 ||resid||/||D||  mean=%.3f  max=%.3f'
          % (float(np.mean(resid_ratio)), float(np.max(resid_ratio))))

    # ---------- 块一：形状 ----------
    print('\n=== 块一：A 曲线形状（本句差分方向，a 以 ||D|| 为单位）===')
    rng_b = np.random.default_rng(SEED + 10)
    curveA, ciA = {}, {}
    for a in ALPHAS:
        m, lo, hi = boot_mean(res_A[a], rng_b)
        curveA[a], ciA[a] = m, (lo, hi)
        print('  a=%.2f  A=%.3f  CI[%.3f,%.3f]  (n=%d)' % (a, m, lo, hi, len(res_A[a])))

    A1 = curveA.get(1.0, float('nan'))
    a_lo = ALPHAS[0]
    a_hi_mid = 0.75 if 0.75 in ALPHAS else ALPHAS[-2]
    mono = all(curveA[ALPHAS[i]] <= curveA[ALPHAS[i + 1]] + 1e-9
               for i in range(len(ALPHAS) - 1))

    if A1 < 0.90:
        shape = 'F0 执行失效（A(1.0)=%.3f < 0.90）：hook/采集有误，整轮作废' % A1
    elif curveA[a_lo] >= 0.90:
        shape = 'F1 早期饱和（A(%.2f)=%.3f）：末位贴近决策边界，细微推动即翻' % (a_lo, curveA[a_lo])
    elif (mono and curveA[a_lo] <= 0.60 and A1 >= 0.90
          and 0.50 <= curveA[a_hi_mid] <= 0.99):
        shape = ('F2 线性可加（单调、A(%.2f)=%.2f、A(%.2f)=%.2f、A(1.0)=%.2f）'
                 % (a_lo, curveA[a_lo], a_hi_mid, curveA[a_hi_mid], A1))
    elif all(curveA[a] <= 0.30 for a in ALPHAS if a <= a_hi_mid) and A1 >= 0.90:
        shape = 'F3 阈值型（a<=%.2f 全 <=0.30，a=1.0 跳到 %.2f）：需完整抵消，无中间态' % (a_hi_mid, A1)
    else:
        shape = 'F4 中间型（未落入 F1/F2/F3，见上表曲线；不得硬塞归类）'
    print('  => %s' % shape)

    # ---------- 块二：方向特异性 ----------
    print('\n=== 块二：方向特异性（A 与 B 同 a 配对差）===')
    best_gap, best_a, best_ci = -1.0, None, (float('nan'), float('nan'))
    gap_tbl = {}
    for a in ALPHAS:
        dm, lo, hi = boot_diff(res_A[a], res_B[a], rng_b)
        gap_tbl[a] = (dm, lo, hi)
        print('  a=%.2f  B=%.3f  A-B=%+.3f  CI[%+.3f,%+.3f]'
              % (a, float(np.mean(res_B[a])), dm, lo, hi))
        if dm > best_gap:
            best_gap, best_a, best_ci = dm, a, (lo, hi)

    if best_gap >= 0.30 and best_ci[0] > 0:
        spec = 'G1 方向特异（max A-B=%+.3f @a=%.2f，CI 下界 %+.3f>0）' % (best_gap, best_a, best_ci[0])
    elif best_gap <= 0.20:
        spec = 'G2 仅幅度效应（max A-B=%+.3f <=0.20）：翻转由扰动幅度决定，与方向无关' % best_gap
    else:
        spec = 'G3 弱特异（max A-B=%+.3f，位于 0.20-0.30）' % best_gap
    print('  => %s' % spec)

    # ---------- 块三：S/R 分解 ----------
    print('\n=== 块三：共享方向 S vs 特异残差 R（a 扫描，同幅度）===')
    rng_b = np.random.default_rng(SEED + 20)
    curveS, curveR = {}, {}
    for a in ALPHAS:
        ms, los, his = boot_mean(res_S[a], rng_b)
        mr, lor, hir = boot_mean(res_R[a], rng_b)
        curveS[a] = ms
        curveR[a] = mr
        print('  a=%.2f  S=%.3f CI[%.3f,%.3f]   R=%.3f CI[%.3f,%.3f]'
              % (a, ms, los, his, mr, lor, hir))

    S1, R1 = curveS[1.0], curveR[1.0]
    A1_v = curveA[1.0]
    if S1 >= 0.70 and R1 <= 0.30:
        decomp = 'H1 共享驱动（S=%.3f >=0.70，R=%.3f <=0.30）：角色被压成跨实体共享的单一方向' % (S1, R1)
    elif S1 <= 0.30 and R1 >= 0.70:
        decomp = 'H2 特异驱动（S=%.3f，R=%.3f）：翻转靠每样本特异分量，共享分量不承载' % (S1, R1)
    elif S1 <= 0.30 and R1 <= 0.30 and A1_v >= 0.90:
        decomp = ('H3 非线性交互（S=%.3f、R=%.3f 单独都不够，A(1.0)=%.3f 够）：'
                  '两分量须同时到位' % (S1, R1, A1_v))
    elif S1 >= 0.70 and R1 >= 0.70:
        decomp = 'H3 双分量冗余（S=%.3f、R=%.3f 均可单独翻转）' % (S1, R1)
    else:
        decomp = 'H4 中间（S=%.3f，R=%.3f，A(1.0)=%.3f；不得硬塞归类）' % (S1, R1, A1_v)
    print('  => %s' % decomp)

    # ---------- 预注册预测对账 ----------
    print('\n=== 预注册预测对账（F2 + G1 + H1）===')
    print('  形状 %s | 特异 %s | 分解 %s' % (shape[:2], spec[:2], decomp[:2]))
    hit = sum([shape.startswith('F2'), spec.startswith('G1'), decomp.startswith('H1')])
    print('  命中 %d/3' % hit)

    verdict = '%s ｜ %s ｜ %s ｜ 预测命中 %d/3' % (shape, spec, decomp, hit)
    print('\nverdict: %s' % verdict)

    payload = {
        'read_layer': READ_LAYER,
        'alphas': list(ALPHAS),
        'n_trip': N_TRIP,
        'forms': list(FORMS),
        'gate': gate,
        'n_pairs': len(recs),
        'cos': {'mean': round(cos_mean, 4), 'median': round(cos_p50, 4),
                'max': round(cos_max, 4), 'n': int(cos_arr.size)},
        'self_cos_lodo': {'mean': round(float(np.mean(self_cos)), 4),
                          'min': round(float(np.min(self_cos)), 4)},
        'resid_ratio': {'mean': round(float(np.mean(resid_ratio)), 4),
                        'max': round(float(np.max(resid_ratio)), 4)},
        'curve_A': {str(a): round(curveA[a], 4) for a in ALPHAS},
        'curve_A_ci': {str(a): [round(ciA[a][0], 4), round(ciA[a][1], 4)] for a in ALPHAS},
        'curve_B': {str(a): round(float(np.mean(res_B[a])), 4) for a in ALPHAS},
        'curve_S': {str(a): round(curveS[a], 4) for a in ALPHAS},
        'curve_R': {str(a): round(curveR[a], 4) for a in ALPHAS},
        'gap_AB': {str(a): [round(gap_tbl[a][0], 4), round(gap_tbl[a][1], 4),
                            round(gap_tbl[a][2], 4)] for a in ALPHAS},
        'shape': shape, 'spec': spec, 'decomp': decomp,
        'pred_hit': int(hit),
        'verdict': verdict,
    }
    p = provenance.write_result('pat_q2c2_direction', payload,
                                model_path=MODEL_PATH, seed=SEED,
                                tool='tools/pat_q2c2_direction.py')
    print('\n落盘 %s' % p)
    print('用时 %.1f 分钟' % ((time.time() - t0) / 60.0))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
