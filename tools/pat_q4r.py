# -*- coding: utf-8 -*-
"""Q4-R：规则型判据首测——不问"角色存在哪里"，问"哪条规则能预测出答案"。

预注册：research/gpt5/docs/PAT_Q4R_PREREG_v1.md
框架：PSF-2（D20–D24），本 Phase 是 §9.3「从定位转向规则+预测」的第一测。

核心约束（本脚本全部难点来源）
------------------------------
规则不得使用答案词，也不得使用"我知道 ag 的答案是 A"这类语言知识。
否则准确率必然 1.0，测到的是我的中文水平，不是模型机制。
三条规则都只输出**哪个槽**，槽里的词由句子本身给出：

  R1  argmax_slot 末位 query 在该槽实体上的注意力质量（完整 s=q·k，RoPE 后自算）
  R2  同 R1，但把三个实体位置的 key 换成去上下文的静态码 k̄(w)
  R3  该句式最后一个实体槽（**不看查询**）—— 位置规则，真备择

实体跨度的现实（SMOKE 暴露的设计修正）
--------------------------------------
12 个人名里只有「大卫」是单 token，其余全是 2 token（张/三）⇒
"槽 = 一个位置"不成立，必须按**词跨度**处理：
  R1 的槽质量 = 注意力在该槽整个跨度上的和
  R2 的静态码按跨度内**逐位置**存 k̄，并要求被测格的跨度长度与夹具一致

防作弊五条
----------
1. 规则不含答案词；2. 层选择按三元组 5 折，l* 只在训练折选；
3. R2 的 k̄ 来自**独立夹具**（不与被测格共用前缀）；4. 自算注意力，不用模型
   output_attentions（bf16 存储误差 4e-3，T1 已测）；5. 同时报全层剖面并注明
   其对层选择是 in-sample。

用法
----
  SMOKE=1 .venv/Scripts/python.exe -X utf8 tools/pat_q4r.py
  .venv/Scripts/python.exe -X utf8 tools/pat_q4r.py
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
M_FIX = 2 if SMOKE else 3
N_SPLIT = 2 if SMOKE else 5

NOUNS = ['苹果', '香蕉', '桃子', '面包', '蛋糕', '杯子', '碗', '毛巾', '雨伞', '铅笔',
         '书包', '钥匙', '手表', '眼镜', '帽子', '围巾', '手套', '鞋子', '镜子', '想法',
         '计划', '建议', '问题', '答案', '理由', '办法', '习惯', '故事', '照片', '礼物']

QUERIES = {'ag': '是谁送的？', 're': '送给了谁？', 'th': '送的是什么？'}
ROLES = ('agent', 'recipient', 'theme')
FORMS = ('A', 'C')
POL = {'agent': 'A', 'recipient': 'B', 'theme': 'C'}


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


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    rng = np.random.default_rng(SEED)
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

    def span_k(ids, span):
        """跑一次，返回该跨度内每个位置的 raw k（未加 RoPE）[n][L,KV,HD]。"""
        fwd(ids)
        return [np.stack([cap[li]['k'][0, p].float().cpu().numpy()
                          for li in range(L)]) for p in span]

    pw = PERSONS if not SMOKE else PERSONS[:4]
    nw = NOUNS if not SMOKE else NOUNS[:6]

    trips = [(pw[i % len(pw)], pw[(i + 3) % len(pw)], nw[i % len(nw)])
             for i in range(N_TRIP)]
    print('三元组 %d，名词互不相同 %d 个' % (len(trips), len({t[2] for t in trips})))

    # ---------- 1 独立夹具：k̄(w, 句式, 槽, 跨度长度, 位置内序号) ----------
    kbar = {}
    n_fix = 0
    for form in FORMS:
        for role in ROLES:
            words = nw if role == 'theme' else pw
            for w in words:
                buf = {}
                for _ in range(M_FIX):
                    a = pw[rng.integers(0, len(pw))]
                    b = pw[rng.integers(0, len(pw))]
                    while b == a and len(pw) > 1:
                        b = pw[rng.integers(0, len(pw))]
                    c = nw[rng.integers(0, len(nw))]
                    fill = {'A': a, 'B': b, 'C': c}
                    fill[POL[role]] = w
                    ctx, rg = build((fill['A'], fill['B'], fill['C']), form)
                    full = render(tok, ctx)
                    base = full.find(ctx)
                    sp, ids = slot_spans(tok, full, rg, base)
                    if not sp[role]:
                        continue
                    buf.setdefault(len(sp[role]), []).append(span_k(ids, sp[role]))
                    n_fix += 1
                for n, lst in buf.items():
                    kbar[(form, role, w, n)] = [
                        np.mean(np.stack([x[i] for x in lst]), axis=0) for i in range(n)]
    print('夹具前向 %d 次，k̄ 覆盖 %d 个 (句式,槽,词,跨度长)' % (n_fix, len(kbar)))
    # 预先搬到 GPU，避免每层每格重复搬运
    kbarT = {k: [torch.tensor(a, device=dev, dtype=torch.float32) for a in v]
             for k, v in kbar.items()}
    print('阶段1 %.0fs' % (time.time() - t0))

    # ---------- 2 144 格：三种规则的预测槽（逐层） ----------
    P1_list, P2_list, P3_list, meta = [], [], [], []
    beh = {q: [0, 0] for q in QUERIES}
    miss_kb = 0

    for ti, trip in enumerate(trips):
        for form in FORMS:
            ctx, rg = build(trip, form)
            last_role = max(rg.items(), key=lambda kv: kv[1][0])[0]
            r3 = np.zeros(3, dtype=np.int16)
            r3[ROLES.index(last_role)] = 1
            for qk, qtext in QUERIES.items():
                user = ctx + qtext
                full = render(tok, user)
                base = full.find(user)
                sp, ids = slot_spans(tok, full, rg, base)
                if any(not sp[r] for r in ROLES):
                    continue
                out = fwd(ids)

                lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
                sc = {x: float(lp[tok(x, add_special_tokens=False)['input_ids'][0]]) for x in trip}
                want = {'ag': trip[0], 're': trip[1], 'th': trip[2]}[qk]
                beh[qk][1] += 1
                beh[qk][0] += int(max(sc, key=sc.get) == want)

                cos, sin = cap['rope']
                cl, sl = cos[0, -1].float(), sin[0, -1].float()
                cT, sT = cos[0].float(), sin[0].float()
                a1 = np.zeros((L, 3), dtype=np.float32)
                a2 = np.zeros((L, 3), dtype=np.float32)
                for li in range(L):
                    qr = rope_vec(cap[li]['q'][0, -1].float(), cl, sl)
                    kr = rope_vec(cap[li]['k'][0].float(), cT[:, None, :], sT[:, None, :])
                    Ac = torch.softmax(
                        torch.einsum('hd,thd->ht', qr, kr.repeat_interleave(rep, dim=1)) * scale,
                        dim=1)
                    for ri, role in enumerate(ROLES):
                        a1[li, ri] = float(Ac[:, sp[role]].sum())
                    # R2：把三个实体的 key 整体换成静态码
                    kr2 = kr.clone()
                    for role in ROLES:
                        kb = kbarT.get((form, role, trip[ROLES.index(role)], len(sp[role])))
                        if kb is None:
                            miss_kb += 1
                            continue
                        for j, p in enumerate(sp[role]):
                            kr2[p] = rope_vec(kb[j][li], cT[p], sT[p])
                    Ac2 = torch.softmax(
                        torch.einsum('hd,thd->ht', qr, kr2.repeat_interleave(rep, dim=1)) * scale,
                        dim=1)
                    for ri, role in enumerate(ROLES):
                        a2[li, ri] = float(Ac2[:, sp[role]].sum())

                p1 = np.zeros((L, 3), dtype=np.int16)
                p2 = np.zeros((L, 3), dtype=np.int16)
                p1[np.arange(L), a1.argmax(1)] = 1
                p2[np.arange(L), a2.argmax(1)] = 1
                P1_list.append(p1)
                P2_list.append(p2)
                P3_list.append(r3)
                ar = [i for i, r in enumerate(ROLES) if trip[i] == want]
                meta.append({'ti': ti, 'form': form, 'qk': qk, 'ans': want,
                             'ans_role': ar[0] if ar else -1,
                             'model_ans': max(sc, key=sc.get),
                             'mass1': a1.copy(), 'mass2': a2.copy()})

    pred1, pred2, pred3 = np.stack(P1_list), np.stack(P2_list), np.stack(P3_list)
    N = len(meta)
    ans_role = np.array([m['ans_role'] for m in meta])
    tids = np.array([m['ti'] for m in meta])
    print('有效格 %d，行为门 %s，静态码缺失 %d'
          % (N, {q: '%d/%d' % tuple(beh[q]) for q in QUERIES}, miss_kb))

    def acc_at(pred, li, ar=None):
        a = ans_role if ar is None else ar
        return float((pred[:, li, :].argmax(1) == a).mean())

    prof1 = np.array([acc_at(pred1, l) for l in range(L)])
    prof2 = np.array([acc_at(pred2, l) for l in range(L)])
    print('R1 逐层准确率（每 3 层）：' + ' '.join('%.2f' % x for x in prof1[::3]))
    print('R2 逐层准确率（每 3 层）：' + ' '.join('%.2f' % x for x in prof2[::3]))
    acc3 = float((pred3.argmax(1) == ans_role).mean())
    print('R3（不看查询）= %.3f' % acc3)

    ut = np.unique(tids)
    ho1, ho2, lstars = [], [], []
    for s in range(N_SPLIT):
        perm = rng.permutation(ut)
        tr = set(perm[:len(perm) // 2].tolist())
        mtr = np.array([t in tr for t in tids])
        if mtr.sum() < 6 or (~mtr).sum() < 6:
            continue
        ar_tr = ans_role[mtr]
        l1 = int(np.argmax([acc_at(pred1[mtr], l, ar_tr) for l in range(L)]))
        l2 = int(np.argmax([acc_at(pred2[mtr], l, ar_tr) for l in range(L)]))
        lstars.append([l1, l2])
        ho1.append(acc_at(pred1[~mtr], l1, ans_role[~mtr]))
        ho2.append(acc_at(pred2[~mtr], l2, ans_role[~mtr]))
    acc1 = float(np.mean(ho1)) if ho1 else float('nan')
    acc2 = float(np.mean(ho2)) if ho2 else float('nan')
    print('held-out：R1 %.3f / R2 %.3f / R3 %.3f；l* = %s' % (acc1, acc2, acc3, lstars))

    l1_glob = int(np.argmax(prof1))
    perm_ok, perm_n = 0, 0
    for ti in ut:
        for form in FORMS:
            idx = [i for i, m in enumerate(meta) if m['ti'] == ti and m['form'] == form]
            if len(idx) != 3:
                continue
            pr = [int(pred1[i, l1_glob].argmax()) for i in idx]
            ar = [int(ans_role[i]) for i in idx]
            perm_n += 1
            perm_ok += int(len(set(pr)) == 3 and pr == ar)
    perm_rate = perm_ok / max(perm_n, 1)
    print('置换一致率 %d/%d = %.3f' % (perm_ok, perm_n, perm_rate))

    def mcnemar(pa, pb):
        ca, cb = (pa == ans_role), (pb == ans_role)
        b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
        if b + c == 0:
            return b, c, 1.0
        from math import comb
        n, k = b + c, min(b, c)
        return b, c, min(2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n), 1.0)

    q1 = pred1[:, l1_glob].argmax(1)
    q2 = pred2[:, int(np.argmax(prof2))].argmax(1)
    q3 = pred3.argmax(1)
    b13, c13, p13 = mcnemar(q1, q3)
    b12, c12, p12 = mcnemar(q1, q2)
    half_ci = 1.96 * float(np.sqrt(max(acc1 * (1 - acc1), 1e-9) / max(N, 1)))
    print('McNemar R1vsR3 b=%d c=%d p=%.2e | R1vsR2 b=%d c=%d p=%.2e' % (b13, c13, p13, b12, c12, p12))
    print('D23 功效 n=%d，acc 95%% 半宽 %.3f' % (N, half_ci))

    # ---- 诊断 1：规则对模型实际输出的准确率（防“规则只是复述输出”）----
    tri_of = {i: t for i, t in enumerate(trips)}
    pred_word = [tri_of[m['ti']][int(pred1[i, l1_glob].argmax())] for i, m in enumerate(meta)]
    acc_model = float(np.mean([pw == m['model_ans'] for pw, m in zip(pred_word, meta)]))

    # ---- 诊断 2：l* 处的分类边际（top1 - top2）/ top1 ----
    mg = []
    for m in meta:
        v = np.sort(m['mass1'][l1_glob])[::-1]
        mg.append(float((v[0] - v[1]) / max(v[0], 1e-9)))
    margin_med = float(np.median(mg))
    margin_min = float(np.min(mg))

    print('诊断：规则 vs 模型实际输出 %.4f（模型自身行为门 %.4f）'
          % (acc_model, np.mean([m['ans'] == m['model_ans'] for m in meta])))
    print('诊断：l*=%d 分类边际 中位 %.3f / 最小 %.3f' % (l1_glob, margin_med, margin_min))

    P1 = bool(acc1 >= 0.80 and acc1 >= 0.3333 + 0.40)
    P2 = bool(acc2 >= acc1 - 0.05)
    P3 = bool((acc1 - acc3) >= 0.30)
    P4 = bool(perm_rate >= 0.80)
    if not P1:
        verdict = 'SKETCH_FALSIFIED' if half_ci < 0.10 else 'pending_low_power'
    elif P3 and P2 and P4:
        verdict = 'RULE_CONFIRMED_STATIC'
    elif P3 and P4:
        verdict = 'RULE_CONFIRMED_CONTEXT'
    else:
        verdict = 'POSITION_RULE'
    print('\n判决 = %s' % verdict)
    print('P1 %.3f %s | P2 %.3f vs %.3f %s | P3 差 %.3f %s | P4 %.3f %s'
          % (acc1, P1, acc2, acc1 - 0.05, P2, acc1 - acc3, P3, perm_rate, P4))

    rec = {
        'seed': SEED, 'smoke': SMOKE, 'n_cells': N, 'n_triples': len(trips),
        'n_fixture_fwd': n_fix, 'm_fixture': M_FIX, 'kbar_entries': len(kbar),
        'behaviour_gate': {q: '%d/%d' % tuple(beh[q]) for q in QUERIES},
        'acc_holdout_R1': round(acc1, 4), 'acc_holdout_R2': round(acc2, 4),
        'acc_R3_position': round(acc3, 4), 'random_baseline': 0.3333,
        'lstar_per_fold': lstars, 'lstar_global_R1': l1_glob,
        'profile_R1': [round(float(x), 4) for x in prof1],
        'profile_R2': [round(float(x), 4) for x in prof2],
        'permutation_rate': round(perm_rate, 4), 'permutation_n': perm_n,
        'mcnemar_R1_R3': {'b': b13, 'c': c13, 'p': p13},
        'mcnemar_R1_R2': {'b': b12, 'c': c12, 'p': p12},
        'ci_half_width_D23': round(half_ci, 4),
        'acc_vs_model_output': round(acc_model, 4),
        'model_own_behaviour': round(float(np.mean([m['ans'] == m['model_ans'] for m in meta])), 4),
        'margin_median_at_lstar': round(margin_med, 4),
        'margin_min_at_lstar': round(margin_min, 4),
        'criteria': {'P1': P1, 'P2': P2, 'P3': P3, 'P4': P4},
        'verdict': verdict,
        'note': '规则只输出槽位、不含答案词；l* 按三元组切分在训练折选；'
                'R2 静态码来自独立夹具；注意力自算不用 output_attentions。',
        'elapsed_s': round(time.time() - t0, 1),
    }
    provenance.write_result('pat_q4r', rec, model_path=MODEL_PATH, seed=SEED, tool='pat_q4r.py')
    print('完成 %.0fs' % (time.time() - t0))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
