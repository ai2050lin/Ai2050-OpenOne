# -*- coding: utf-8 -*-
"""Q2-C1 位置修补：角色信息在哪个位置、哪一层被搬运。

为什么不是原预注册的"主动/被动"
--------------------------------
原设计"把被动句 agent 位置的激活换成主动句的"没有可测效应：
主动句与被动句答案相同（都是张三），换与不换输出都不该变。D6 情形，已在执行前废弃。

改用标准因果追踪配对：同句式、实体互换，唯一差别是"谁是 agent"，答案相反。
  clean   = 张三送给了赵六一个苹果。是谁送的？  -> 张三
  corrupt = 赵六送给了张三一个苹果。是谁送的？  -> 赵六
探查已验证：两种句式下 T 均 39/39 等长，实体 token 索引完全对齐，
行为门 8/8 全对且 log P 差 11–26 nats（tools/pat_q2c1_probe.py）。

干预
----
在 corrupt 运行中，把第 i 层输出、位置 p 的残差流替换为 clean 在同一位置的值。
位置 p ∈ {agent 首 token, agent 末 token, recipient 首, theme 首, 问句首, 末位}。

主指标 = **答案翻转率**：patch 后答 agent（clean 的施事）的比例。
  corrupt 基线 0，clean 1。翻转率天然有界于 [0,1]，直接对应行为。
  （不用 restore：(patch后ratio - corrupt) / (clean - corrupt)，
   SMOKE 显示它能到 1.44，因为把"张六"这种怪词搬进了 corrupt，
   值域超出 clean，不可按 0-1 阈值判读。）

判据：形状分类（预注册，互斥，四种覆盖完备）
--------------------------------------------
执行前必须承认：**patch 名字位置在早期层几乎必然"有效"**，
因为那时该位置的表示基本就是 token 嵌入，换过去等于换了个词。
所以不能预设"从低升到高"，否则一定误判。

  S1 双峰（可定位迁移）：agent 位置翻转率早期 >=0.80、中段跌到 <=0.40，
     且末位翻转率在中段之后升到 >=0.60
     => 信息从名字位置迁到末位；转折层 L* = agent 位置首次跌破 0.50 的层。
  S2 早饱和（设计失效）：agent 位置翻转率全程 >=0.80 且末位全程 <=0.30
     => 只测到嵌入复制，本设计无法定位迁移层，结论作废，须换方法。
  S3 全程无效：所有位置翻转率全程 <=0.40
     => 单点 patch 搬不动该信息，须改集合 patch 或改用方向干预。
  S4 末位主导：末位翻转率全程 >=0.60 而 agent 位置 <=0.40
     => 每层都已汇聚到末位，与 S2 相反，同样不可定位迁移点。

表层复制对照（区分"角色信息"与"token 传送带"）
----------------------------------------------
若 agent 位置生效只是因为"把答案词本身搬过去"，那么换成**第三个实体**
（既非 agent 也非 recipient）的激活时，输出应当变成那个第三实体。
每层额外跑一次 third-patch，记 third_hit = patch 后 argmax 为第三实体的比例：
  third_hit 高 => 该位置只是答案词的传送带（表层复制），不是角色绑定
  third_hit 低 => 该位置携带角色绑定，换词也不改"谁是 agent"

用法
----
  SMOKE=1 .venv/Scripts/python.exe -X utf8 tools/pat_q2c1_patch.py
  .venv/Scripts/python.exe -X utf8 tools/pat_q2c1_patch.py
"""
from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / 'tools')):
    if p not in sys.path:
        sys.path.insert(0, p)

import provenance
from pat_q1_unbind import PERSONS, QUERIES, build, encode, render, triples

SEED = 20261110
SMOKE = os.environ.get('SMOKE', '') == '1'
N_TRIP = 4 if SMOKE else 24
FORMS = ('A', 'C')
LAYER_STEP = 12 if SMOKE else 1
POS_NAMES = ('agent0', 'agent1', 'recip0', 'theme0', 'q0', 'last')


def build_rec(tok, trip, form):
    ctx, ranges = build(trip, form)
    raw = ctx + QUERIES['agent']
    text = render(tok, raw)
    ids, idx = encode(tok, text, raw, ranges)
    return {'ids': ids, 'idx': idx, 'trip': trip, 'ctx_len': len(ctx),
            'text': text, 'raw': raw}


def pos_map(tok, rec):
    agent = rec['idx']['agent']
    pm = {'agent0': agent[0], 'agent1': agent[-1],
          'recip0': rec['idx']['recipient'][0],
          'theme0': rec['idx']['theme'][0],
          'last': len(rec['ids']) - 1}
    enc = tok(rec['text'], return_offsets_mapping=True, add_special_tokens=False)
    base = rec['text'].find(rec['raw'])
    q_char = base + rec['ctx_len']
    q_tok = [i for i, (a, b) in enumerate(enc['offset_mapping']) if a <= q_char < b]
    pm['q0'] = q_tok[0] if q_tok else len(rec['ids']) - 2
    return pm


def pick_third(tok, trip):
    """选一个 token 数与 agent 相同的第三实体（既非 agent 也非 recipient）。"""
    a_len = len(tok(trip[0], add_special_tokens=False)['input_ids'])
    for cand in PERSONS:
        if cand in (trip[0], trip[1]):
            continue
        if len(tok(cand, add_special_tokens=False)['input_ids']) == a_len:
            return cand
    return None


@contextmanager
def patch_layer(layer, pos, vec):
    def fn(module, inputs, output):
        if isinstance(output, tuple):
            hs = output[0].clone()
            hs[0, pos, :] = vec
            return (hs,) + tuple(output[1:])
        hs = output.clone()
        hs[0, pos, :] = vec
        return hs

    h = layer.register_forward_hook(fn)
    try:
        yield
    finally:
        h.remove()


@torch.no_grad()
def last_lp(model, dev, ids):
    o = model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
    return torch.log_softmax(o.logits[0, -1].float(), dim=-1)


def main() -> int:
    from server import live_analysis_service as las

    t0 = time.time()
    las._load_async()
    while las._state['status'] in ('idle', 'loading'):
        if time.time() - t0 > 600:
            raise RuntimeError('加载超时')
        time.sleep(2.0)
    model, tok = las._state['model'], las._state['tokenizer']
    dev = next(model.parameters()).device
    layers = model.model.layers
    lay_idx = list(range(0, len(layers), LAYER_STEP))

    trips = triples(N_TRIP)
    print('三元组 %d  句式 %s  层 %d 个（step=%d）  位置 %d 个'
          % (N_TRIP, '/'.join(FORMS), len(lay_idx), LAYER_STEP, len(POS_NAMES)))

    flip = {}      # (form, pos, layer) -> list of 0/1
    rest = {}      # 同键，restore 参考值
    third_hit = {} # layer -> list of 0/1
    gate = {'clean_ok': 0, 'corrupt_ok': 0, 'n': 0, 'third_skip': 0}

    for form in FORMS:
        for trip in trips:
            clean = build_rec(tok, trip, form)
            corrupt = build_rec(tok, (trip[1], trip[0], trip[2]), form)
            if len(clean['ids']) != len(corrupt['ids']):
                print('  !! 长度不等，跳过 %s %s' % (form, trip))
                continue
            pm = pos_map(tok, clean)
            ta = tok(trip[0], add_special_tokens=False)['input_ids'][0]
            tb = tok(trip[1], add_special_tokens=False)['input_ids'][0]

            with torch.no_grad():
                oc = model(input_ids=torch.tensor([clean['ids']], device=dev),
                           use_cache=False, output_hidden_states=True)
                lp_c = torch.log_softmax(oc.logits[0, -1].float(), dim=-1)
                r_clean = float(lp_c[ta]) - float(lp_c[tb])
                gate['clean_ok'] += int(lp_c[ta] > lp_c[tb])
                lp_x = last_lp(model, dev, corrupt['ids'])
                r_cor = float(lp_x[ta]) - float(lp_x[tb])
                gate['corrupt_ok'] += int(lp_x[tb] > lp_x[ta])
                gate['n'] += 1
            denom = r_clean - r_cor
            hs = oc.hidden_states

            third = pick_third(tok, trip)
            third_rec = build_rec(tok, (third, trip[1], trip[2]), form) if third else None
            third_pos = third_rec['idx']['agent'][0] if third_rec else None
            third_id = tok(third, add_special_tokens=False)['input_ids'][0] if third else None
            oc3 = None
            if third_rec and len(third_rec['ids']) == len(clean['ids']):
                with torch.no_grad():
                    oc3 = model(input_ids=torch.tensor([third_rec['ids']], device=dev),
                                use_cache=False, output_hidden_states=True)
            else:
                gate['third_skip'] += 1

            for L in lay_idx:
                for pn in POS_NAMES:
                    p = pm[pn]
                    vec = hs[L + 1][0, p, :].clone()
                    with torch.no_grad(), patch_layer(layers[L], p, vec):
                        op = last_lp(model, dev, corrupt['ids'])
                    fl = int(int(op.argmax()) == ta)
                    flip.setdefault((form, pn, L), []).append(fl)
                    if abs(denom) > 1e-6:
                        rest.setdefault((form, pn, L), []).append(
                            (float(op[ta]) - float(op[tb]) - r_cor) / denom)
                    if oc3 is not None and pn == 'agent0':
                        vec3 = oc3.hidden_states[L + 1][0, third_pos, :].clone()
                        with torch.no_grad(), patch_layer(layers[L], p, vec3):
                            op3 = last_lp(model, dev, corrupt['ids'])
                        third_hit.setdefault(L, []).append(int(int(op3.argmax()) == third_id))

    print('行为门：clean %d/%d 答 agent；corrupt %d/%d 答 recipient；third 不可用 %d 次'
          % (gate['clean_ok'], gate['n'], gate['corrupt_ok'], gate['n'], gate['third_skip']))

    def med(d, form, pn, L):
        v = d.get((form, pn, L))
        return float(np.mean(v)) if v else float('nan')

    print('\n=== 翻转率层剖面 ===')
    for form in FORMS:
        print('句式 %s' % form)
        print('  层  ' + ''.join('%-9s' % p for p in POS_NAMES) + 'third@agent0')
        for L in lay_idx:
            cells = ''.join('%.2f     ' % med(flip, form, pn, L) for pn in POS_NAMES)
            th = float(np.mean(third_hit[L])) if L in third_hit else float('nan')
            print('  %2d  %s%.2f' % (L, cells, th))

    def band(form, pn, lo, hi, src):
        vs = [med(src, form, pn, L) for L in lay_idx if lo <= L <= hi]
        vs = [v for v in vs if not np.isnan(v)]
        return float(np.mean(vs)) if vs else float('nan')

    third_n = max(1, len(lay_idx) // 3)
    early = (lay_idx[0], lay_idx[third_n - 1])
    mid = (lay_idx[third_n], lay_idx[2 * third_n - 1])
    late = (lay_idx[2 * third_n], lay_idx[-1])

    print('\n=== 分带翻转率（早 %s / 中 %s / 晚 %s）===' % (early, mid, late))
    bandv = {}
    for form in FORMS:
        for pn in POS_NAMES:
            e = band(form, pn, *early, src=flip)
            m = band(form, pn, *mid, src=flip)
            l = band(form, pn, *late, src=flip)
            bandv[(form, pn)] = (e, m, l)
            print('  %s %-7s 早=%.2f 中=%.2f 晚=%.2f' % (form, pn, e, m, l))
    th_e = np.nanmean([float(np.mean(third_hit[L])) for L in lay_idx if early[0] <= L <= early[1]] or [np.nan])
    th_m = np.nanmean([float(np.mean(third_hit[L])) for L in lay_idx if mid[0] <= L <= mid[1]] or [np.nan])
    th_l = np.nanmean([float(np.mean(third_hit[L])) for L in lay_idx if late[0] <= L <= late[1]] or [np.nan])
    print('  third@agent0  早=%.2f 中=%.2f 晚=%.2f' % (th_e, th_m, th_l))

    notes = []
    for form in FORMS:
        a_e = max(bandv[(form, pn)][0] for pn in ('agent0', 'agent1'))
        a_m = min(bandv[(form, pn)][1] for pn in ('agent0', 'agent1'))
        a_l = min(bandv[(form, pn)][2] for pn in ('agent0', 'agent1'))
        l_e, l_m, l_l = bandv[(form, 'last')]
        if a_e >= 0.80 and a_m <= 0.40 and l_l >= 0.60:
            notes.append('%s:S1 双峰' % form)
        elif a_e >= 0.80 and l_l <= 0.30:
            notes.append('%s:S2 早饱和' % form)
        elif max(a_e, a_m, a_l, l_l) <= 0.40:
            notes.append('%s:S3 全程无效' % form)
        elif l_l >= 0.60 and a_e <= 0.40:
            notes.append('%s:S4 末位主导' % form)
        else:
            notes.append('%s:未归类(早%.2f 中%.2f 晚%.2f 末%.2f)' % (form, a_e, a_m, a_l, l_l))
    v = ' | '.join(notes)
    print('\n=== 判读（形状分类）===')
    print('  ' + v)

    thy = float(np.nanmean([float(np.mean(vv)) for vv in third_hit.values()] or [np.nan]))
    if thy >= 0.60:
        tv = 'T-表面伪影：agent0 patch 的效应主要是答案词的传送带（third_hit=%.2f）' % thy
    elif thy <= 0.20:
        tv = 'T-角色绑定：换第三实体后仍答原 agent（third_hit=%.2f），该位置携带角色而非词' % thy
    else:
        tv = 'T-混合：third_hit=%.2f，表层与角色成分并存' % thy
    print('  ' + tv)

    drop = {}
    for form in FORMS:
        for pn in ('agent0', 'agent1'):
            for L in lay_idx:
                if med(flip, form, pn, L) <= 0.50:
                    drop['%s/%s' % (form, pn)] = L
                    break
    print('  首次跌破 0.50 的层：%s' % (drop or '无'))

    prof = {}
    for form in FORMS:
        for pn in POS_NAMES:
            prof['%s|%s' % (form, pn)] = [
                {'layer': L,
                 'flip': round(med(flip, form, pn, L), 4),
                 'restore_med': round(float(np.median(rest[(form, pn, L)])), 4)
                 if (form, pn, L) in rest else None}
                for L in lay_idx if (form, pn, L) in flip]

    p = provenance.write_result(
        'pat_q2c1_patch',
        {'n_trip': N_TRIP, 'forms': list(FORMS), 'layers': lay_idx,
         'pos_names': list(POS_NAMES), 'gate': gate,
         'band_flip': {'%s|%s' % k: [round(x, 4) for x in vv] for k, vv in bandv.items()},
         'third_hit_by_band': [round(float(th_e), 4), round(float(th_m), 4), round(float(th_l), 4)],
         'third_hit_overall': round(float(thy), 4),
         'profiles': prof, 'first_drop_below_050': drop,
         'verdict': v, 'verdict_third': tv},
        seed=SEED, tool='tools/pat_q2c1_patch.py')
    print('\n落盘 %s' % p)
    print('用时 %.1f 分钟' % ((time.time() - t0) / 60.0))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
