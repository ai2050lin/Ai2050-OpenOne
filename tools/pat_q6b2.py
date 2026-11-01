# -*- coding: utf-8 -*-
"""Q6-B2：A2 的负值是几何必然，还是语义反排列。

预注册：research/gpt5/docs/PAT_Q6B2_PREREG_v1.md（判据冻结，不得改阈值）

为什么要这一测
--------------
A2 的 cos(Δ(a→b), Δ(b→c)) = −0.246，但两跳**共享中间词 b 且符号相反**。
对独立同分布的各向同性向量，E[cos] = −σ²/(2σ²) = **−0.5**。
故负值可能只是几何后果，与层级无关；跨链对照 = 0 不能判定同链负值的来源。

三个竞争解释（判决集互斥）：G 几何必然 / S 上位词区域 / H 层级延伸

两个零假设共用同一估计器（只换 c）：
  null_rand  ← 从全词表均匀抽 d
  null_level ← 从上位词池抽 d（针对解释 S 的层级匹配对照）

用法
----
  SMOKE=1 .venv/Scripts/python.exe -X utf8 tools/pat_q6b2.py
  .venv/Scripts/python.exe -X utf8 tools/pat_q6b2.py
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
from pat_q6a import FAM, CHAINS, TMPL, SLOT, MODEL_PATH, SEED, word_span

SMOKE = os.environ.get('SMOKE', '') == '1'
N_NULL = 40 if SMOKE else 200
N_BOOT = 200 if SMOKE else 1000

SUP = ['食物', '生物', '物品', '蔬菜', '水果', '动物']

# 判据（冻结）
G0_LO, G0_HI = -0.65, -0.35      # null_rand 全层中位必须落在区间内
TH_GEOM = 0.05                   # |obs − null| < 0.05 视为与几何相容
TH_DIFF = 0.10                   # 显著差异阈值（= MDE 下方，D23 已声明）
TH_FRAC = 0.30
TH_FRAC_GEOM = 0.70
TH_SHARED = 0.30                  # Q6-A 的 TH_W：已确立共享方向的量级


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    rng = np.random.default_rng(SEED)
    dev = 'cuda'

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, attn_implementation='eager').to(dev).eval()
    NL = model.config.num_hidden_layers + 1
    H = model.config.hidden_size

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
    for w in SUP:
        if w not in vocab:
            vocab.append(w)
    widx = {w: i for i, w in enumerate(vocab)}
    sup_idx = [widx[w] for w in SUP if w in widx]
    print('词表 %d（含上位词池 %d）；链 %d' % (len(vocab), len(sup_idx), len(CHAINS)))

    tmpl = TMPL[:1] if SMOKE else TMPL
    V = np.zeros((len(vocab), NL, H), dtype=np.float32)
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
            V[wi] = acc / got if got else 0.0
    print('表征 %.0fs' % (time.time() - t0))

    chains = [(widx[a], widx[b], widx[c]) for a, b, c in CHAINS
              if a in widx and b in widx and c in widx]
    print('可用链 %d' % len(chains))
    nv = len(vocab)

    prof_obs = np.zeros(NL)
    prof_nr = np.zeros(NL)
    prof_nl = np.zeros(NL)
    prof_t = np.zeros(NL)
    prof_t_null = np.zeros(NL)
    prof_rho = np.zeros(NL)
    per_chain = np.zeros((len(chains), NL))     # obs − null_rand，供 bootstrap

    # 预先抽好 null 索引，保证逐层同一批（减少层间抖动）
    null_rand_idx = np.stack([rng.integers(0, nv - 1, size=(len(chains), N_NULL))
                              for _ in range(1)])[0]
    null_lev_idx = np.stack([rng.integers(0, len(sup_idx), size=(len(chains), N_NULL))
                             for _ in range(1)])[0]

    for li in range(NL):
        obs, nr, nl, ts, tn, rh, pc = [], [], [], [], [], [], []
        for ci, (a, b, c) in enumerate(chains):
            va, vb, vc = V[a, li], V[b, li], V[c, li]
            h1, h2 = vb - va, vc - vb
            n1 = np.linalg.norm(h1) + 1e-12
            u = h1 / n1
            o = float(h2 @ u / (np.linalg.norm(h2) + 1e-12))
            obs.append(o)
            ts.append(float((h2 @ u) / n1))
            rh.append(float(np.linalg.norm(h2 - (h2 @ u) * u) / (np.linalg.norm(h2) + 1e-12)))

            # null_rand：抽 d，排除 a/b/c
            cand = rng.integers(0, nv, size=N_NULL * 2)
            cand = np.array([x for x in cand if x not in (a, b, c)][:N_NULL])
            if len(cand) < N_NULL:
                cand = np.array([x for x in range(nv) if x not in (a, b, c)])
            hd = V[cand, li] - vb
            nr.append(float(np.median(hd @ u / (np.linalg.norm(hd, axis=1) + 1e-12))))
            tn.append(float(np.median((hd @ u) / n1)))

            cl = rng.integers(0, len(sup_idx), size=N_NULL)
            cd = np.array([sup_idx[x] for x in cl])
            cd = cd[(cd != a) & (cd != b) & (cd != c)]
            if len(cd) >= 5:
                hd2 = V[cd, li] - vb
                nl.append(float(np.median(hd2 @ u / (np.linalg.norm(hd2, axis=1) + 1e-12))))
            else:
                nl.append(np.nan)
            pc.append(o - nr[-1])
        prof_obs[li] = float(np.median(obs))
        prof_nr[li] = float(np.median(nr))
        prof_nl[li] = float(np.nanmedian(nl))
        prof_t[li] = float(np.median(ts))
        prof_t_null[li] = float(np.median(tn))
        prof_rho[li] = float(np.median(rh))
        per_chain[:, li] = np.array(pc)

    # ---------- 参照线：Q6-A 已确立的共享方向长什么样（§6.3） ----------
    ref = {}
    for f in ('F3', 'F4'):
        pairs = [(widx[a], widx[b]) for a, b in FAM[f] if a in widx and b in widx]
        prof = np.zeros(NL)
        for li in range(NL):
            D = unit(np.stack([V[b, li] - V[a, li] for a, b in pairs]))
            pp = [float(D[i] @ D[j]) for i in range(len(pairs)) for j in range(i + 1, len(pairs))
                  if pairs[i][1] != pairs[j][1]]
            prof[li] = float(np.median(pp)) if pp else np.nan
        ref[f] = prof

    # ---------- G0 前置闸 ----------
    g0 = float(np.median(prof_nr))
    gate = G0_LO <= g0 <= G0_HI
    print('\n=== G0 框架前置闸 ===')
    print('null_rand 全层中位 %.3f，要求落在 [%.2f, %.2f] ⇒ %s'
          % (g0, G0_LO, G0_HI, '通过' if gate else '未过（判 FRAMEWORK_INVALID）'))

    print('\n=== 逐层剖面（D25：37 层全打印，每行 12 点同一层序）===')
    for name, arr, fmt in (('obs      ', prof_obs, '%+.2f'), ('null_rand', prof_nr, '%+.2f'),
                           ('null_lev ', prof_nl, '%+.2f'),
                           ('obs−rand ', prof_obs - prof_nr, '%+.2f'),
                           ('obs−lev  ', prof_obs - prof_nl, '%+.2f'),
                           ('t        ', prof_t, '%.2f'),
                           ('t_null   ', prof_t_null, '%.2f'),
                           ('rho      ', prof_rho, '%.2f')):
        print('%s %s' % (name, ' '.join(fmt % x for x in arr)))

    self_max = float(np.max(prof_obs))
    difr = prof_obs - prof_nr
    dfl = prof_obs - prof_nl
    f_geom = float(np.mean(np.abs(difr) < TH_GEOM))
    f_pos = float(np.mean(difr >= TH_DIFF))
    f_neg = float(np.mean(difr <= -TH_DIFF))
    f_lvl = float(np.mean(np.abs(dfl) < TH_GEOM)) if not np.all(np.isnan(dfl)) else 0.0
    print('\n层占比：|obs−rand|<%.2f %.2f；obs−rand≥+%.2f %.2f；≤−%.2f %.2f；'
          '|obs−lev|<%.2f %.2f'
          % (TH_GEOM, f_geom, TH_DIFF, f_pos, TH_DIFF, f_neg, TH_GEOM, f_lvl))

    # ---------- bootstrap（重采样单位 = 链；峰值层 = |obs−rand| 中位最小处不适用，改用 |difr| 最大层之外的判决层） ----------
    lpeak = int(np.argmax(np.abs(difr)))
    x = per_chain[:, lpeak]
    boots = np.array([np.median(rng.choice(x, len(x), replace=True)) for _ in range(N_BOOT)])
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    ok_ci = ci[0] <= float(np.median(x)) <= ci[1]
    print('判决层 L%d（|obs−rand| 最大）：obs−rand 中位 %+.3f  CI [%+.3f, %+.3f]  '
          '包住点估计=%s' % (lpeak, float(np.median(x)), ci[0], ci[1], ok_ci))
    if not ok_ci:
        print('!! ESTIMATOR_BUG（D26）：CI 不含点估计')
        main, attr = 'ESTIMATOR_BUG', 'NOT_APPLICABLE'
        note = 'bootstrap 口径与点估计不一致，须先修（D26）'
    elif not gate:
        main = 'FRAMEWORK_INVALID'
        attr = 'NOT_APPLICABLE'
        note = 'null_rand 全层中位 %.3f 不在 [%.2f,%.2f]，零假设构造有误' % (g0, G0_LO, G0_HI)
    else:
        main = 'SHARED_DIRECTION' if float(np.mean(prof_obs >= TH_SHARED)) >= TH_FRAC \
            else 'NOT_A_SHARED_DIRECTION'
        dfl_ok = not np.all(np.isnan(dfl))
        if f_geom >= TH_FRAC_GEOM and ci[0] <= 0 <= ci[1]:
            attr = 'PURE_GEOMETRY'
            note = ('%.0f%% 的层 |obs−null_rand|<%.2f 且判决层 CI 含 0 ⇒ '
                    '负值与「共享中间词 b」的几何后果不可区分' % (f_geom * 100, TH_GEOM))
        elif f_pos >= TH_FRAC and ci[0] > 0 and (not dfl_ok or f_lvl < TH_FRAC):
            attr = 'WEAK_ABOVE_GEOMETRY'
            note = ('obs 显著高于几何零假设（+%.2f）但绝对水平仅 %.2f，'
                    '远低于已确立共享方向的 %.2f ⇒ 弱对齐，不是方向'
                    % (float(np.median(difr)), float(self_max), TH_SHARED))
        elif f_pos >= TH_FRAC and f_lvl >= TH_FRAC:
            attr = 'SUPERORDINATE_EXPLAINED'
            note = 'obs 与 null_level 无差异、与 null_rand 有差异 ⇒ 对齐全由上位词区域解释'
        elif f_neg >= TH_FRAC and ci[1] < 0:
            attr = 'ANTI_BEYOND_GEOMETRY'
            note = '%.0f%% 的层 obs−rand ≤ −%.2f 且 CI 上界 <0 ⇒ 反向排列超出几何' % (f_neg * 100, TH_DIFF)
        else:
            attr = 'ATTRIBUTION_UNCLEAR'
            note = '各档皆不满足；n=10 链的 MDE≈0.14，只能记 pending'

    t_frac = float(np.mean(prof_t > 1.0))
    t_frac_null = float(np.mean(prof_t_null > 1.0))
    print('\n主判 = %s（obs 最高 %.2f，共享方向阈值 %.2f）' % (main, self_max, TH_SHARED))
    print('归因 = %s' % attr)
    print('说明：%s' % note)
    print('参照线（Q6-A 已确立的共享方向）：F3 颜色中位 %.2f / F4 大小中位 %.2f'
          % (float(np.nanmedian(ref['F3'])), float(np.nanmedian(ref['F4']))))
    print('t>1 层占比：obs %.2f vs null %.2f（层级延伸若成立，obs 应更高）'
          % (t_frac, t_frac_null))

    payload = {
        'phase': 'Q6-B2', 'n_chains': len(chains), 'n_vocab': nv,
        'n_null': N_NULL, 'n_templates': len(tmpl),
        'G0_null_rand_median': round(g0, 4), 'G0_gate': bool(gate),
        'profile_obs': [round(float(x), 4) for x in prof_obs],
        'profile_null_rand': [round(float(x), 4) for x in prof_nr],
        'profile_null_level': [None if np.isnan(x) else round(float(x), 4) for x in prof_nl],
        'profile_diff_rand': [round(float(x), 4) for x in difr],
        'profile_diff_level': [None if np.isnan(x) else round(float(x), 4) for x in dfl],
        'profile_t': [round(float(x), 4) for x in prof_t],
        'profile_t_null': [round(float(x), 4) for x in prof_t_null],
        'profile_rho': [round(float(x), 4) for x in prof_rho],
        'frac_geom': round(f_geom, 4), 'frac_pos': round(f_pos, 4),
        'frac_neg': round(f_neg, 4), 'frac_level_match': round(f_lvl, 4),
        'decision_layer': lpeak, 'diff_CI': [round(ci[0], 4), round(ci[1], 4)],
        'ci_contains_point': bool(ok_ci),
        't_gt1_frac_obs': round(t_frac, 4), 't_gt1_frac_null': round(t_frac_null, 4),
        'median_obs_all_layers': round(float(np.median(prof_obs)), 4),
        'ref_within_F3': [round(float(x), 4) for x in ref['F3']],
        'ref_within_F4': [round(float(x), 4) for x in ref['F4']],
        'ref_F3_median': round(float(np.nanmedian(ref['F3'])), 4),
        'ref_F4_median': round(float(np.nanmedian(ref['F4'])), 4),
        'obs_max': round(self_max, 4),
        'main_verdict': main, 'attribution': attr, 'verdict': '%s / %s' % (main, attr),
        'note': note,
        'elapsed_s': round(time.time() - t0, 1),
    }
    out = provenance.write_result('pat_q6b2', payload, model_path=MODEL_PATH,
                                  seed=SEED, tool='tools/pat_q6b2.py')
    print('\n落盘 %s' % out)
    print('%s / %s  用时 %.0fs' % (main, attr, time.time() - t0))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
