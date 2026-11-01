#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Q2-C2c: 同长度四方向对照 —— 读出对「方向偏离」如何定价？

为什么必须补这一轮
----------------
Q2-C2b（res 042214）把三种方向对齐到相同的「沿 Δ 投影比 p」，
但只有 A 与 R 的位移长度同时相等（都 = p*||Δ||）：
  A: 位移 = p*Δ_i，        |位移| = p*n，Δ 投影 = p*n
  R: 位移 = (p/sin)*resid，|位移| = p*n，Δ 投影 = p*n
  S: 位移 = (p/cos)*n*m̂，  |位移| = p*n/cos  <-- 长度失控，cos=0.482 时是 2.07 倍
故 S 曲线非单调（p=0.75 0.639 -> p=1.0 0.611）是**过冲**，非方向特权。
K2 的判决因此不可靠，必须用长度受控的对照重做。

设计
----
固定位移长度 L = p*||Δ_i||，只变方向，四组：

  D 本句差分   单位向量 Δ̂_i           与 Δ 夹角  0°
  R 特异残差   单位向量 r̂_i = resid/|resid|   夹角 arccos(sin_i) ≈ 33°
  M 留一共识   单位向量 m̂_i(-i)        夹角 arccos(cos_i)  ≈ 61°
  Z 随机       单位随机向量             夹角 ≈ 90°（期望）

p ∈ {0.5, 0.75, 1.0}。四组 |位移| 严格相等，唯一变量是方向。

预注册判据（执行前写死）
------------------
  M0 基线：p=0 不干预时四组翻转率均 0.000
  M1 方向特异：D - Z >= 0.30（同长度下方向决定翻转，非幅度）
  M2 本句特异：D - M >= 0.20（本句差分优于跨样本共识方向）
  M3 角度连续定价：翻转率随「与 Δ 的夹角」单调下降（D >= R >= M >= Z，容差 0.10）
     => 读出方向 = 本句 Δ 方向，对偏离连续敏感（可给出角度-效率曲线）
  M4 一维投影：四组在等长度下无差异（所有差 <=0.15）
     => 只看投影、不看方向（与 M1/M3 互斥）

预注册预测：M1 + M2 + M3 成立（本句方向最好、共识次之、随机最差，连续下降）
  —— 若成立，则 Q2-C2b 的 K2「多方向」应改述为「单一但样本特异的读出方向」，
     而非「存在多个独立方向」。这一区分决定图谱形态：
     前者 = 每个样本一个方向（无共享轴），后者 = 多个可命名的方向通道。

成本：48 样本 x 3 p x 4 方向 + 96 采集 ≈ 670 次前向。
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
from pat_q2c1_patch import build_rec, patch_layer, pos_map
from pat_q1_unbind import triples

SEED = 20261113
SMOKE = os.environ.get('SMOKE', '') == '1'
N_TRIP = 24
PS = (0.5, 1.0) if SMOKE else (0.5, 0.75, 1.0)
FORMS = ('A', 'C')
READ_LAYER = 33
N_BOOT = 5000
MODEL_PATH = 'D:/develop/model/Qwen/Qwen3-4B'


def boot_mean(vals, rng):
    a = np.asarray(vals, dtype=float)
    if a.size == 0:
        return float('nan'), float('nan'), float('nan')
    idx = rng.integers(0, a.size, size=(N_BOOT, a.size))
    m = a[idx].mean(axis=1)
    return float(a.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_diff_dict(da, db, rng):
    keys = sorted(set(da) & set(db))
    if not keys:
        return float('nan'), float('nan'), float('nan'), 0
    d = np.array([da[k] - db[k] for k in keys], dtype=float)
    idx = rng.integers(0, d.size, size=(N_BOOT, d.size))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), len(keys)


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

    print('=== Q2-C2c 同长度四方向对照 ===')
    print('freeze: READ_LAYER=%d  p=%s  n_trip=%d  四组 |位移| 严格相等' % (READ_LAYER, PS, N_TRIP))

    trips = triples(N_TRIP)
    recs, gate = [], {'clean_ok': 0, 'corrupt_ok': 0, 'n': 0}
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
    print('行为门：clean %d/%d；corrupt %d/%d' % (gate['clean_ok'], gate['n'],
                                                 gate['corrupt_ok'], gate['n']))

    all_delta = torch.stack([r['delta'] for r in recs])
    m_hat = {}
    for k in range(len(recs)):
        others = torch.cat([all_delta[:k], all_delta[k + 1:]], dim=0).mean(dim=0)
        m_hat[k] = others / max(float(others.norm()), 1e-9)

    rng_dir = np.random.default_rng(SEED + 1)
    res = {g: {p: {} for p in PS} for g in ('D', 'R', 'M', 'Z')}
    base = {}
    ang = {'D': [], 'R': [], 'M': [], 'Z': []}
    per_sample = []

    for k, r in enumerate(recs):
        hx, d, dt = r['hx'], r['delta'], r['dt']
        nrm = float(d.norm())
        if nrm < 1e-6:
            continue
        pos, ids, ta, layer = r['last'], r['ids'], r['ta'], layers[READ_LAYER]
        uD = d / nrm
        m = m_hat[k]
        cos_i = float(torch.dot(uD, m))
        resid = d - cos_i * nrm * m
        uM = m
        uR = resid / max(float(resid.norm()), 1e-9)
        vz = rng_dir.standard_normal(d.shape[0]).astype(np.float32)
        vz = vz / max(float(np.linalg.norm(vz)), 1e-9)
        uZ = torch.from_numpy(vz).to(hx.device)
        # 实测夹角（与 Δ 单位方向）
        for nm, u in (('D', uD), ('R', uR), ('M', uM), ('Z', uZ)):
            ang[nm].append(float(torch.dot(u, uD)))

        # M0：p=0 基线
        base[k] = int(predict(model, dev, ids, pos, hx.to(dt), layer) == ta)

        for p in PS:
            L = p * nrm
            for nm, u in (('D', uD), ('R', uR), ('M', uM), ('Z', uZ)):
                v = (hx + L * u).to(dt)
                res[nm][p][k] = int(predict(model, dev, ids, pos, v, layer) == ta)

        per_sample.append({
            'k': k, 'form': r['form'], 'cos_M': round(cos_i, 4),
            'cos_R': round(float(torch.dot(uR, uD)), 4),
            'flip': {('%s@p%s' % (nm, p)): res[nm][p].get(k)
                     for nm in ('D', 'R', 'M', 'Z') for p in PS}})

    print('实测夹角余弦（与 Δ̂）：' + '  '.join(
        '%s=%.3f' % (nm, float(np.mean(ang[nm]))) for nm in ('D', 'R', 'M', 'Z')))

    rng_b = np.random.default_rng(SEED + 20)
    bm, blo, bhi = boot_mean(list(base.values()), rng_b)
    print('\n=== M0：p=0 不干预基线 = %.3f CI[%.3f,%.3f]（应 0.000）' % (bm, blo, bhi))

    print('\n=== 同长度四方向（|位移| = p*||Δ||）===')
    print('   p      D 本句      R 残差      M 共识      Z 随机')
    cur = {}
    for p in PS:
        row = []
        for nm in ('D', 'R', 'M', 'Z'):
            m_, lo, hi = boot_mean(list(res[nm][p].values()), rng_b)
            cur[(nm, p)] = (m_, lo, hi)
            row.append(m_)
        print('  %.2f  %.3f       %.3f       %.3f       %.3f'
              % (p, row[0], row[1], row[2], row[3]))

    print('\n=== 配对差（同 p、同样本）===')
    dDZ, dDM, dRM, dMR = {}, {}, {}, {}
    for p in PS:
        a = boot_diff_dict(res['D'][p], res['Z'][p], rng_b)
        b = boot_diff_dict(res['D'][p], res['M'][p], rng_b)
        c = boot_diff_dict(res['R'][p], res['M'][p], rng_b)
        e = boot_diff_dict(res['M'][p], res['Z'][p], rng_b)
        dDZ[p], dDM[p], dRM[p], dMR[p] = a, b, c, e
        print('  p=%.2f  D-Z=%+.3f CI[%+.3f,%+.3f]   D-M=%+.3f CI[%+.3f,%+.3f]   '
              'R-M=%+.3f   M-Z=%+.3f'
              % (p, a[0], a[1], a[2], b[0], b[1], b[2], c[0], e[0]))

    m1 = all(dDZ[p][0] >= 0.30 for p in PS)
    m2 = all(dDM[p][0] >= 0.20 for p in PS)
    mono, mono_detail = True, []
    for p in PS:
        seq = [cur[(nm, p)][0] for nm in ('D', 'R', 'M', 'Z')]
        ok = all(seq[i] >= seq[i + 1] - 0.10 for i in range(3))
        mono = mono and ok
        mono_detail.append({'p': p, 'seq': [round(x, 4) for x in seq], 'monotone': bool(ok)})
    max_abs_all = max(abs(cur[(a_, p)][0] - cur[(b_, p)][0])
                      for p in PS for i, a_ in enumerate(('D', 'R', 'M', 'Z'))
                      for b_ in ('D', 'R', 'M', 'Z') if a_ != b_)
    m4 = max_abs_all <= 0.15

    print('\n=== 判据 ===')
    print('  M1 方向特异（D-Z 最小 %+.3f >=0.30）：%s'
          % (min(dDZ[p][0] for p in PS), '成立' if m1 else '不成立'))
    print('  M2 本句特异（D-M 最小 %+.3f >=0.20）：%s'
          % (min(dDM[p][0] for p in PS), '成立' if m2 else '不成立'))
    print('  M3 角度连续定价（D>=R>=M>=Z，容差 0.10）：%s'
          % ('成立' if mono else '不成立'))
    for d_ in mono_detail:
        print('     p=%.2f  %s  %s' % (d_['p'], d_['seq'], 'ok' if d_['monotone'] else 'X'))
    print('  M4 一维投影（最大组间差 %.3f <=0.15）：%s'
          % (max_abs_all, '成立' if m4 else '不成立'))

    # ---------- 分层：M 组封顶是不是 cos 异质造成的混合 ----------
    print('\n=== 分层：按 cos(D̂, M̂) 分箱（M 组是否只在近邻样本上有效）===')
    strata = []
    coss = np.array([s['cos_M'] for s in per_sample], dtype=float)
    for lo, hi in ((0.0, 0.3), (0.3, 0.6), (0.6, 0.8), (0.8, 1.01)):
        sel = [s for s in per_sample if lo <= s['cos_M'] < hi]
        if not sel:
            continue
        row = {'bin': [lo, hi], 'n': len(sel)}
        for nm in ('D', 'R', 'M'):
            v = float(np.mean([s['flip']['%s@p1.0' % nm] for s in sel]))
            row[nm] = round(v, 4)
        strata.append(row)
        print('  cos∈[%.1f,%.1f)  n=%2d   D@1.0=%.3f  R@1.0=%.3f  M@1.0=%.3f'
              % (lo, hi, len(sel), row['D'], row['R'], row['M']))
    mflip = np.array([s['flip']['M@p1.0'] for s in per_sample], dtype=float)
    corr_M = float(np.corrcoef(coss, mflip)[0, 1]) if coss.size > 2 else float('nan')
    print('  corr(cos_M, M@1.0) = %+.3f  （>0 则 M 的有效性由近邻程度决定，非固定特权）'
          % corr_M)

    if m4:
        verdict = 'M4 一维投影：同长度下方向无差异，读出只看投影（M1/M2/M3 均不成立）'
    else:
        parts = ['M1 %s' % ('方向特异' if m1 else '方向不特异'),
                 'M2 %s' % ('本句特异' if m2 else '无本句特异'),
                 'M3 %s' % ('角度连续定价' if mono else '角度不单调')]
        verdict = ' ｜ '.join(parts) + '（最大组间差 %.3f）' % max_abs_all
    print('\nverdict: %s' % verdict)

    payload = {
        'read_layer': READ_LAYER, 'ps': list(PS), 'n_pairs': len(recs), 'gate': gate,
        'ang_cos': {nm: round(float(np.mean(ang[nm])), 4) for nm in ('D', 'R', 'M', 'Z')},
        'base': [round(bm, 4), round(blo, 4), round(bhi, 4)],
        'curves': {('%s@p%s' % (nm, p)): [round(cur[(nm, p)][0], 4),
                                          round(cur[(nm, p)][1], 4),
                                          round(cur[(nm, p)][2], 4)]
                   for nm in ('D', 'R', 'M', 'Z') for p in PS},
        'diff': {('D-Z@p%s' % p): [round(x, 4) for x in dDZ[p]] for p in PS},
        'diff_DM': {('D-M@p%s' % p): [round(x, 4) for x in dDM[p]] for p in PS},
        'mono_detail': mono_detail,
        'strata_by_cos': strata,
        'corr_cos_M': round(corr_M, 4),
        'per_sample': per_sample,
        'M0': bool(bm <= 0.02), 'M1': bool(m1), 'M2': bool(m2),
        'M3': bool(mono), 'M4': bool(m4),
        'verdict': verdict,
    }
    pth = provenance.write_result('pat_q2c2c_samelength', payload,
                                  model_path=MODEL_PATH, seed=SEED,
                                  tool='tools/pat_q2c2c_samelength.py')
    print('\n落盘 %s' % pth)
    print('用时 %.1f 分钟' % ((time.time() - t0) / 60.0))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
