# -*- coding: utf-8 -*-
"""Q2-L2  词法翻转是「读出点效应」还是「线性读出的平凡后果」

Q2-L 结论：L33 末位注入纯嵌入差 unit(E[X]−E[Y])，翻转率 1.000；
剥掉 Δ 的该分量后降到 0.159（随机方向 0.091）。判为 O1「全词法」。

但这个结论有一个必须排除的平凡解释：**tied embedding + 末层 RMSNorm 之后
直接与 E 做内积**，那么「往残差流里加 E[X]−E[Y] 就抬高 X 的 logit」几乎是
代数恒等式，不是机制发现。若如此，Q2-L 的 O1 只是测量装置的镜像。

两个判决性控制
--------------
  P（位置控制）同一向量 unit(E[X]−E[Y])，位移同为 p·||d1||，
      注入位置改为 **问句首 token q0**（非末位）。
      · 仍翻 ⇒ 与末位无关，是全局 logit 注入 ⇒ O1 平凡
      · 不翻 ⇒ 末位是特殊读出位 ⇒ O1 有意义

  L（层控制）同一向量、同一末位，注入层扫 {L20, L25, L29, L31, L33, L35}。
      · 只在末段生效 ⇒ 前面层会把注入洗掉，末位是读出汇聚点 ⇒ 有意义
      · 早期层同样生效 ⇒ 注入直通输出，不承载信息流 ⇒ 平凡

  附加 T（词表控制）：注入 unit(E[theme0]−E[Y])，看是否答成 theme。
      若任意词都能注入成功，说明末位是个通用「下一词转向器」，
      与「角色解绑」无关，进一步支持平凡解释。

判据（预注册）
--------------
  T1 平凡：P 位置注入 >= 0.70 或 L 控制在 L20 已 >= 0.70
  T2 读出点特殊：P <= 0.30 且 L20 <= 0.30 且 L35 >= 0.70
  T3 中间：其余
  门：末位 L33 基线需 >= 0.80，否则整轮作废。

每样本逐条命中一律落盘（D5：不做无分布的均值）。

成本：44 样本 × (1 采集 + 8 注入) ≈ 400 次前向。
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
from pat_q2c1_patch import build_rec, patch_layer, pos_map
from pat_q2l_decompose import build_triples, pick_new_person, unitize

SEED = 20261115
SMOKE = os.environ.get('SMOKE', '') == '1'
N_TRIP = 4 if SMOKE else 24
FORMS = ('A', 'C')
READ_LAYER = 33
LAYERS = (20, 25, 29, 31, 33, 35)
PS = (1.0,) if SMOKE else (0.75, 1.0)
N_BOOT = 5000
MODEL_PATH = 'D:/develop/model/Qwen/Qwen3-4B'


def boot_mean(vals, rng):
    a = np.asarray(vals, dtype=float)
    if a.size == 0:
        return float('nan'), float('nan'), float('nan')
    idx = rng.integers(0, a.size, size=(N_BOOT, a.size))
    m = a[idx].mean(axis=1)
    return float(a.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


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

    print('=== Q2-L2  词法翻转：读出点效应 vs 平凡线性读出 ===')
    print('freeze: READ_LAYER=%d  LAYERS=%s  p=%s  n_trip=%d'
          % (READ_LAYER, LAYERS, PS, N_TRIP))

    trips = build_triples(N_TRIP)
    recs, gate = [], {'n': 0, 'b_ok': 0, 'p1_ok': 0}
    for form in FORMS:
        for trip in trips:
            X, Y, T = trip
            Yp = pick_new_person(tok, [Y, X])
            if Yp is None:
                continue
            B = build_rec(tok, (Y, X, T), form)
            P1 = build_rec(tok, (X, Y, T), form)
            if len(B['ids']) != len(P1['ids']):
                continue
            pm = pos_map(tok, B)
            tY = tok(Y, add_special_tokens=False)['input_ids'][0]
            tX = tok(X, add_special_tokens=False)['input_ids'][0]
            tT = tok(T, add_special_tokens=False)['input_ids'][0]
            hB, lpB = collect_last(model, dev, B['ids'])
            h1, lp1 = collect_last(model, dev, P1['ids'])
            gate['n'] += 1
            gate['b_ok'] += int(float(lpB[tY]) > float(lpB[tX]))
            gate['p1_ok'] += int(float(lp1[tX]) > float(lp1[tY]))
            d1 = h1.float() - hB.float()
            recs.append({'form': form, 'trip': trip, 'ids': B['ids'],
                         'dt': hB.dtype, 'hB': hB.float(), 'n1': float(d1.norm()),
                         'q0': pm['q0'], 'last': len(B['ids']) - 1,
                         'lex1': unitize((emb[tX] - emb[tY]).float().detach()),
                         'lexT': unitize((emb[tT] - emb[tY]).float().detach()),
                         'tX': tX, 'tY': tY, 'tT': tT, 'd1': d1})

    print('行为门：B->Y %d/%d  P1->X %d/%d' % (gate['b_ok'], gate['n'], gate['p1_ok'], gate['n']))
    if not recs:
        print('无可分析样本，退出')
        return 1

    res = {('last@%d' % L): {p: {} for p in PS} for L in LAYERS}
    res_pos = {p: {} for p in PS}
    res_t = {p: {} for p in PS}

    for k, r in enumerate(recs):
        dt, ids = r['dt'], r['ids']
        for p in PS:
            L0 = p * r['n1']
            # 层控制：同一向量、同一末位、不同层
            for L in LAYERS:
                v = (r['hB'] + L0 * r['lex1']).to(dt)
                res[('last@%d' % L)][p][k] = int(
                    predict(model, dev, ids, r['last'], v, layers[L]) == r['tX'])
            # 位置控制：同一向量、同一层 L33、注入问句首
            v = (r['hB'] + L0 * r['lex1']).to(dt)
            res_pos[p][k] = int(
                predict(model, dev, ids, r['q0'], v, layers[READ_LAYER]) == r['tX'])
            # 词表控制：注入 theme 的嵌入差
            v = (r['hB'] + L0 * r['lexT']).to(dt)
            res_t[p][k] = int(
                predict(model, dev, ids, r['last'], v, layers[READ_LAYER]) == r['tT'])

    rng_b = np.random.default_rng(SEED + 20)
    print('\n=== 层控制：同一词法向量注入末位，扫注入层 ===')
    cur = {}
    for p in PS:
        line = []
        for L in LAYERS:
            m_, lo, hi = boot_mean(list(res[('last@%d' % L)][p].values()), rng_b)
            cur[(L, p)] = (m_, lo, hi)
            line.append('L%d=%.3f' % (L, m_))
        print('  p=%.2f  ' % p + '  '.join(line))

    print('\n=== 位置控制：注入问句首 q0（L33）===')
    pos_m = {}
    for p in PS:
        m_, lo, hi = boot_mean(list(res_pos[p].values()), rng_b)
        pos_m[p] = m_
        print('  p=%.2f  命中 %.3f  CI[%.3f,%.3f]' % (p, m_, lo, hi))

    print('\n=== 词表控制：注入 theme 的嵌入差（末位 L33）===')
    t_m = {}
    for p in PS:
        m_, lo, hi = boot_mean(list(res_t[p].values()), rng_b)
        t_m[p] = m_
        print('  p=%.2f  答成 theme %.3f  CI[%.3f,%.3f]' % (p, m_, lo, hi))

    P = PS[-1]
    base_L33 = cur[(READ_LAYER, P)][0]
    p_pos = pos_m[P]
    l20 = cur[(20, P)][0]
    l35 = cur[(35, P)][0]

    e0 = base_L33 >= 0.80
    if p_pos >= 0.70 or l20 >= 0.70:
        code, desc = 'T1', '平凡：注入直通输出，与末位读出点无关'
    elif p_pos <= 0.30 and l20 <= 0.30 and l35 >= 0.70:
        code, desc = 'T2', '读出点特殊：仅末段末位生效'
    else:
        code, desc = 'T3', '中间：部分位点/层特异'

    print('\n=== 判据 ===')
    print('  门（末位 L33 @p=%.2f = %.3f >=0.80）: %s'
          % (P, base_L33, 'PASS' if e0 else 'FAIL 整轮作废'))
    print('  位置控制 q0 = %.3f ｜ L20 = %.3f ｜ L35 = %.3f' % (p_pos, l20, l35))
    print('  => %s %s' % (code, desc))
    print('  词表控制：注入 theme 嵌入差 → 答成 theme %.3f' % t_m[P])

    verdict = '%s %s（q0=%.3f L20=%.3f L35=%.3f，L33=%.3f）｜theme 注入 %.3f｜门 %s' % (
        code, desc, p_pos, l20, l35, base_L33, t_m[P], 'PASS' if e0 else 'FAIL')
    print('\nverdict: %s' % verdict)

    payload = {
        'read_layer': READ_LAYER, 'layers': list(LAYERS), 'ps': list(PS),
        'n_trip': N_TRIP, 'n_samples': len(recs), 'gate': gate,
        'curve_layer': {'L%d@p%.2f' % (L, p): [round(cur[(L, p)][0], 4),
                                                round(cur[(L, p)][1], 4),
                                                round(cur[(L, p)][2], 4)]
                        for L in LAYERS for p in PS},
        'pos_q0': {'p%.2f' % p: round(pos_m[p], 4) for p in PS},
        'lex_theme': {'p%.2f' % p: round(t_m[p], 4) for p in PS},
        'raw': {'last@%d' % L: {str(k): v for k, v in
                                res[('last@%d' % L)][P].items()} for L in LAYERS},
        'raw_q0': {str(k): v for k, v in res_pos[P].items()},
        'verdict': verdict,
    }
    path = provenance.write_result('pat_q2l2_controls', payload,
                                   model_path=MODEL_PATH, seed=SEED,
                                   tool='tools/pat_q2l2_controls.py')
    print('落盘 %s' % path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
