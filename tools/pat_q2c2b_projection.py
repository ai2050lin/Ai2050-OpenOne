#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Q2-C2b: 一维读出器检验 —— 翻转率只由「沿 Δ 的投影」决定吗？

为什么必须做这一轮
----------------
Q2-C2（res 041841 / 待入账）在 L33 末位做向量平移，得到：
  A 本句差分 Δ_i        : 0.000 / 0.292 / 0.854 / 1.000   （a = 0.25/0.5/0.75/1.0）
  B 随机方向            : 全程 0.000
  S 共享方向 m̂(留一)    : 0.000 / 0.021 / 0.458 / 0.479
  R 特异残差 resid      : 0.000 / 0.000 / 0.312 / 0.604

**但 S 与 R 的差别有一个致命的替代解释**：三者幅度都按 ||Δ|| 设定，
而它们在 Δ 方向上的投影并不相同。
  cos(D_i, m̂) = 0.482 ⇒ S 走满一步只前进 0.482 步
  resid 在 Δ 上投影系数 = sqrt(1-cos^2) = 0.876 ⇒ R 走满一步前进 0.876 步
若翻转率只取决于投影，则 S 应弱于 R —— 实测 S(1.0)=0.479 < R(1.0)=0.604，方向一致。
故「S/R 各自承载一半」这一读数**尚未排除幅度混杂，不得作为结论**。

设计
----
对每个样本，把四种方向**对齐到相同的目标投影比 p**（p 以 ||Δ|| 为单位）：

  A: vec = h_x + p * Δ_i                      （因子 1）
  S: vec = h_x + (p / cos_i) * ||Δ|| * m̂_i    （因子 cos_i）
  R: vec = h_x + (p / sin_i) * resid_i        （因子 sin_i = sqrt(1-cos_i^2)）
  B: vec = h_x + 1.0 * ||Δ|| * u_rand         （无法对齐，实测其 p_eff ≈ 0，作对照）

p ∈ {0, 0.25, 0.5, 0.75, 1.0}。p=0 为不干预基线（应翻 0）。
三种对齐方向在**同一 p** 下直接可比，幅度混杂被消除。

预注册判据（互斥，执行前写死）
--------------------------
  K1 p=0 基线复核：所有方向在 p=0 的翻转率 = 0.000（否则 hook 有副作用，作废）
  K2 一维读出器：同 p 下 A/S/R 三方向的翻转率两两差 <= 0.15（全部 p）
     => 翻转只由沿 Δ_i 的投影决定，与方向来源无关。末位是一个线性标量读出器。
  K3 共享方向特权：存在 p 使 S - A >= 0.20（S 在同等投影下更强）
     => 除投影外，方向本身有额外效力（共享方向更"典型"）。
  K4 特异分量不足：存在 p 使 R - A <= -0.20（R 在同等投影下更弱）
     => resid 垂直于 m̂ 使其部分失效，方向结构不止一维。
  K5 幅度效应：B 在 p_eff≈0 时翻转率 >= 0.30
     => 翻转与方向无关，仅由扰动幅度驱动（与 Q2-C2 的 G1 结论冲突，须重查）。

预注册预测：K1 通过 + K2 成立 + K5 不成立
  —— 即「末位读出是一维线性标量」：任何方向只要有足够的沿 Δ 投影都能翻转，
     方向来源（本句/共享/残差）不影响。若成立，Q2-C2 的 H4「中间型」应被重读为
     **幅度未对齐**所致，S/R 各半的说法撤回。

成本：48 样本 x (5 p x 3 方向 + 1 随机) + 96 采集 ≈ 880 次前向。
"""
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
from pat_q2c1_patch import build_rec, patch_layer, pos_map, last_lp
from pat_q1_unbind import triples

SEED = 20261112
SMOKE = os.environ.get('SMOKE', '') == '1'
N_TRIP = 24 if not SMOKE else 24     # 注意：Q2-C2 的 SMOKE 用 4 个三元组只覆盖同一对人物，
                                     # cos 虚高到 0.936，与全量 0.245 完全不符。
                                     # 本轮 SMOKE 仍用全量三元组，只削减 p 网格。
PS = (0.0, 0.5, 1.0) if SMOKE else (0.0, 0.25, 0.5, 0.75, 1.0)
FORMS = ('A', 'C')
READ_LAYER = 33
N_BOOT = 5000
MODEL_PATH = 'D:/develop/model/Qwen/Qwen3-4B'
COS_MIN = 0.10          # S 组要求 cos 下限，避免 1/cos 爆炸
SCALE_CAP = 6.0         # |介入位移| / ||Δ|| 的上限，超出跳过并计数


def boot_diff_dict(da, db, rng):
    """按样本键取交集后做配对差 da-db。da/db 形如 {k: 0/1}。"""
    keys = sorted(set(da) & set(db))
    if not keys:
        return float('nan'), float('nan'), float('nan'), 0
    a = np.array([da[k] for k in keys], dtype=float)
    b = np.array([db[k] for k in keys], dtype=float)
    d = a - b
    idx = rng.integers(0, d.size, size=(N_BOOT, d.size))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), len(keys)


def boot_mean(vals, rng):
    a = np.asarray(vals, dtype=float)
    if a.size == 0:
        return float('nan'), float('nan'), float('nan')
    idx = rng.integers(0, a.size, size=(N_BOOT, a.size))
    m = a[idx].mean(axis=1)
    return float(a.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_diff(a, b, rng):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    if d.size == 0:
        return float('nan'), float('nan'), float('nan')
    idx = rng.integers(0, d.size, size=(N_BOOT, d.size))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


@torch.no_grad()
def collect(model, dev, ids):
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

    print('=== Q2-C2b 一维读出器检验 ===')
    print('freeze: READ_LAYER=%d  p网格=%s  n_trip=%d  COS_MIN=%.2f  SCALE_CAP=%.1f'
          % (READ_LAYER, PS, N_TRIP, COS_MIN, SCALE_CAP))

    trips = triples(N_TRIP)
    recs = []
    gate = {'clean_ok': 0, 'corrupt_ok': 0, 'n': 0}
    for form in FORMS:
        for trip in trips:
            clean = build_rec(tok, trip, form)
            corrupt = build_rec(tok, (trip[1], trip[0], trip[2]), form)
            if len(clean['ids']) != len(corrupt['ids']):
                continue
            pm = pos_map(tok, clean)
            ta = tok(trip[0], add_special_tokens=False)['input_ids'][0]
            tb = tok(trip[1], add_special_tokens=False)['input_ids'][0]
            hc, lp_c = collect(model, dev, clean['ids'])
            hx, lp_x = collect(model, dev, corrupt['ids'])
            gate['clean_ok'] += int(float(lp_c[ta]) > float(lp_c[tb]))
            gate['corrupt_ok'] += int(float(lp_x[tb]) > float(lp_x[ta]))
            gate['n'] += 1
            recs.append({'form': form, 'ids': corrupt['ids'], 'last': pm['last'],
                         'ta': ta, 'dt': hc.dtype,
                         'hx': hx.float(), 'delta': hc.float() - hx.float()})

    print('行为门：clean %d/%d 答 agent；corrupt %d/%d 答 recipient'
          % (gate['clean_ok'], gate['n'], gate['corrupt_ok'], gate['n']))
    print('可用配对 n=%d' % len(recs))

    # 留一共享方向
    all_delta = torch.stack([r['delta'] for r in recs])
    m_hat = {}
    for k in range(len(recs)):
        others = torch.cat([all_delta[:k], all_delta[k + 1:]], dim=0).mean(dim=0)
        m_hat[k] = others / max(float(others.norm()), 1e-9)

    rng_dir = np.random.default_rng(SEED + 1)
    res = {'A': {p: {} for p in PS}, 'S': {p: {} for p in PS},
           'R': {p: {} for p in PS}, 'B': {}}
    cos_list, sin_list, p_eff_B = [], [], []
    skip = {'S_cos': 0, 'S_cap': 0, 'R_cap': 0}

    for k, r in enumerate(recs):
        hx, d, dt = r['hx'], r['delta'], r['dt']
        nrm = float(d.norm())
        if nrm < 1e-6:
            continue
        pos, ids, ta, layer = r['last'], r['ids'], r['ta'], layers[READ_LAYER]
        m = m_hat[k]
        cos_i = float(torch.dot(d, m)) / nrm
        sin_i = float(np.sqrt(max(1.0 - cos_i * cos_i, 1e-8)))
        resid = d - cos_i * nrm * m
        cos_list.append(cos_i)
        sin_list.append(sin_i)

        for p in PS:
            vA = (hx + p * d).to(dt)
            res['A'][p][k] = int(predict(model, dev, ids, pos, vA, layer) == ta)

            if cos_i >= COS_MIN and (p / cos_i) <= SCALE_CAP:
                vS = (hx + (p / cos_i) * nrm * m).to(dt)
                res['S'][p][k] = int(predict(model, dev, ids, pos, vS, layer) == ta)
            elif cos_i < COS_MIN:
                skip['S_cos'] += 1
            else:
                skip['S_cap'] += 1

            if (p / sin_i) <= SCALE_CAP:
                vR = (hx + (p / sin_i) * resid).to(dt)
                res['R'][p][k] = int(predict(model, dev, ids, pos, vR, layer) == ta)
            else:
                skip['R_cap'] += 1

        # B 随机方向：p_eff 实测
        u = rng_dir.standard_normal(d.shape[0]).astype(np.float32)
        u = u / max(float(np.linalg.norm(u)), 1e-9)
        ut = torch.from_numpy(u).to(hx.device)
        p_eff_B.append(float(torch.dot(ut, d)) / nrm)
        vB = (hx + nrm * ut).to(dt)
        res['B'][k] = int(predict(model, dev, ids, pos, vB, layer) == ta)

    print('cos(D,m̂) mean=%.3f   sin mean=%.3f   B 组 p_eff mean=%+.3f |mean|=%.4f'
          % (float(np.mean(cos_list)), float(np.mean(sin_list)),
             float(np.mean(p_eff_B)), float(np.mean(np.abs(p_eff_B)))))
    print('跳过：S(cos<%.2f) %d 次；S(scale cap) %d 次；R(scale cap) %d 次'
          % (COS_MIN, skip['S_cos'], skip['S_cap'], skip['R_cap']))

    rng_b = np.random.default_rng(SEED + 20)
    print('\n=== 逐方向 p 曲线（p = 沿 Δ 的投影比）===')
    print('   p      A              S              R           n(A/S/R)')
    curves = {}
    for p in PS:
        row = {}
        for g in ('A', 'S', 'R'):
            m_, lo, hi = boot_mean(list(res[g][p].values()), rng_b)
            curves[(g, p)] = (m_, lo, hi)
            row[g] = m_
        print('  %.2f  %.3f          %.3f          %.3f       %d/%d/%d'
              % (p, row['A'], row['S'], row['R'],
                 len(res['A'][p]), len(res['S'][p]), len(res['R'][p])))

    bm, blo, bhi = boot_mean(list(res['B'].values()), rng_b)
    print('  B 随机方向（p_eff≈0）：翻转率 %.3f  CI[%.3f,%.3f]' % (bm, blo, bhi))

    # K1
    k1 = all(curves[(g, 0.0)][0] <= 0.02 for g in ('A', 'S', 'R'))
    print('\n=== 判据 ===')
    print('  K1 p=0 基线复核：%s（A=%.3f S=%.3f R=%.3f）'
          % ('通过' if k1 else '失败——hook 有副作用，整轮作废',
             curves[('A', 0.0)][0], curves[('S', 0.0)][0], curves[('R', 0.0)][0]))

    # K2/K3/K4：同 p 配对差
    max_abs_dAS, max_abs_dAR, min_dSA, min_dRA = 0.0, 0.0, 1.0, 1.0
    detail = []
    for p in PS:
        if p == 0.0:
            continue
        dSA, lo_sa, hi_sa, n_sa = boot_diff_dict(res['S'][p], res['A'][p], rng_b)
        dRA, lo_ra, hi_ra, n_ra = boot_diff_dict(res['R'][p], res['A'][p], rng_b)
        max_abs_dAS = max(max_abs_dAS, abs(dSA))
        max_abs_dAR = max(max_abs_dAR, abs(dRA))
        min_dSA = min(min_dSA, dSA)
        min_dRA = min(min_dRA, dRA)
        detail.append({'p': p, 'S_minus_A': [round(dSA, 4), round(lo_sa, 4), round(hi_sa, 4)],
                       'R_minus_A': [round(dRA, 4), round(lo_ra, 4), round(hi_ra, 4)],
                       'n_paired': [n_sa, n_ra]})
        print('  p=%.2f  S-A=%+.3f CI[%+.3f,%+.3f]   R-A=%+.3f CI[%+.3f,%+.3f]'
              % (p, dSA, lo_sa, hi_sa, dRA, lo_ra, hi_ra))

    k2 = (max_abs_dAS <= 0.15 and max_abs_dAR <= 0.15)
    k3 = (max((d['S_minus_A'][0] for d in detail), default=0.0) >= 0.20)
    k4 = (min((d['R_minus_A'][0] for d in detail), default=0.0) <= -0.20)
    k5 = (bm >= 0.30 and float(np.mean(np.abs(p_eff_B))) <= 0.10)

    v_k2 = 'K2 一维读出器成立（三方向最大同 p 差 %.3f <=0.15）：翻转只由沿 Δ 的投影决定' % max(max_abs_dAS, max_abs_dAR) if k2 else \
           'K2 不成立（最大同 p 差 %.3f >0.15）：投影不足以解释，方向来源有独立效应' % max(max_abs_dAS, max_abs_dAR)
    v_k3 = 'K3 共享方向特权（S-A 最大 %+.3f >=0.20）' % max((d['S_minus_A'][0] for d in detail), default=0.0) if k3 else \
           'K3 无特权（S-A 最大 %+.3f <0.20）' % max((d['S_minus_A'][0] for d in detail), default=0.0)
    v_k4 = 'K4 特异分量不足（R-A 最小 %+.3f <=-0.20）' % min((d['R_minus_A'][0] for d in detail), default=0.0) if k4 else \
           'K4 特异分量无不足（R-A 最小 %+.3f >-0.20）' % min((d['R_minus_A'][0] for d in detail), default=0.0)
    v_k5 = 'K5 幅度效应（B 翻 %.3f，p_eff≈%.3f）：与方向无关' % (bm, float(np.mean(np.abs(p_eff_B)))) if k5 else \
           'K5 无幅度效应（B 翻 %.3f，p_eff≈%.3f）：方向特异' % (bm, float(np.mean(np.abs(p_eff_B))))

    for v in (v_k2, v_k3, v_k4, v_k5):
        print('  ' + v)

    if not k1:
        verdict = 'K1 失败，整轮作废'
    elif k2 and not k3 and not k4:
        verdict = 'K2 单方向线性读出（%s；%s；%s；%s）' % (v_k2, v_k3, v_k4, v_k5)
    elif k2 and k3:
        verdict = 'K2+K3 一维为主但共享方向有特权（%s；%s；%s；%s）' % (v_k2, v_k3, v_k4, v_k5)
    else:
        verdict = 'K2 失败：多方向/非线性（%s；%s；%s；%s）' % (v_k2, v_k3, v_k4, v_k5)
    print('\nverdict: %s' % verdict)

    payload = {
        'read_layer': READ_LAYER, 'ps': list(PS), 'n_pairs': len(recs),
        'gate': gate, 'skip': skip,
        'cos_mean': round(float(np.mean(cos_list)), 4),
        'sin_mean': round(float(np.mean(sin_list)), 4),
        'p_eff_B': {'mean_signed': round(float(np.mean(p_eff_B)), 4),
                    'mean_abs': round(float(np.mean(np.abs(p_eff_B))), 4)},
        'curves': {('%s@p%s' % (g, p)): [round(curves[(g, p)][0], 4),
                                         round(curves[(g, p)][1], 4),
                                         round(curves[(g, p)][2], 4)]
                   for g in ('A', 'S', 'R') for p in PS},
        'B': [round(bm, 4), round(blo, 4), round(bhi, 4)],
        'diff_detail': detail,
        'K1': bool(k1), 'K2': bool(k2), 'K3': bool(k3), 'K4': bool(k4), 'K5': bool(k5),
        'verdict': verdict,
    }
    pth = provenance.write_result('pat_q2c2b_projection', payload,
                                  model_path=MODEL_PATH, seed=SEED,
                                  tool='tools/pat_q2c2b_projection.py')
    print('\n落盘 %s' % pth)
    print('用时 %.1f 分钟' % ((time.time() - t0) / 60.0))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
