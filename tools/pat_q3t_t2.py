# -*- coding: utf-8 -*-
"""Q3-T2：注意力分数按「钥匙的身份分量 / 上下文分量」做精确分解。

预注册：research/gpt5/docs/PAT_Q3T_PREREG_v1.md（§2）

为什么这一步值得做
------------------
前面九轮全部死在同一类混杂上：消融死于覆盖率（Q2-A），注入死于线性读出（Q2-L），
移植死于匹配（Q2-L 第一版）。这个测量一个都不碰——
它不改模型、不加向量、不跨上下文搬运，只是把**同一次前向里的一个点积拆成两项**：

    s = q·k = q·k_id + q·k_ctx        k_id 沿身份方向投影，k_ctx 是精确残差

两处与预注册的偏离（跑前记下）
------------------------------
偏离 1（判据）：预注册写 ρ = |corr(s_id, s_act)| / |corr(s_ctx, s_act)|。
    但 s_act = s_id + s_ctx 是恒等式，两个相关系数必然同号，ρ 会系统性失真。
    ⇒ 改用**方差份额** share = var(该项)/var(s_act)，并报协方差项
      （若两项强相关，share 之和会明显偏离 1，那时分解不干净，结论要打折）。
      ρ 并列报告，只作对照。
偏离 2（范围）：身份方向 û 必须在**与目标句完全同构、同位置**的模板上测
    （RoPE 使位置一变方向就转）。所以本轮只分解 **theme 槽** 的 key。
    agent/recipient 槽需要各自的模板族，留给下一轮，本轮不对其作任何主张。

判据（跑前冻结）
----------------
J1 分界层    share_id 层剖面跨中位翻转：前 1/3 中位 > 0.5 且 后 1/3 中位 < 0.2（或反向）
J2 层内分工  同一层 32 个头的 share_id 呈双峰（|mean−median|/std > 0.3）
J3 等权      share_id 全层中位 ∈ [0.35, 0.65]，无翻转、无双峰
三判据互斥，任一成立即判决；全不成立记 NULL。

预测（预注册）：存在分界层，落在 L20–L33。

用法
----
  SMOKE=1 .venv/Scripts/python.exe -X utf8 tools/pat_q3t_t2.py
  .venv/Scripts/python.exe -X utf8 tools/pat_q3t_t2.py
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
from pat_q1_unbind import PERSONS, SYS, build, render

MODEL_PATH = 'D:/develop/model/Qwen/Qwen3-4B'
SEED = 20261101
SMOKE = os.environ.get('SMOKE', '') == '1'
N_CTX = 4 if SMOKE else 20

NOUNS = ['苹果', '香蕉', '桃子', '面包', '蛋糕', '杯子', '碗', '毛巾', '雨伞', '铅笔',
         '书包', '钥匙', '手表', '眼镜', '帽子', '围巾', '手套', '鞋子', '镜子', '想法',
         '计划', '建议', '问题', '答案', '理由', '办法', '习惯', '故事', '照片', '礼物']

QUERIES = {'ag': '是谁送的？', 're': '送给了谁？', 'th': '送的是什么？'}
FORMS = ('A', 'C')
TMPL_UID = {'A': '{A}送给了{B}一个{W}。', 'C': '{W}被{A}送给了{B}。'}


def rope_vec(v, cos_t, sin_t):
    """对 [..., HD] 施加 RoPE，与 transformers 的 rotate_half 约定一致。"""
    hd = v.shape[-1] // 2
    x1, x2 = v[..., :hd], v[..., hd:]
    rot = torch.cat([-x2, x1], dim=-1)
    return v * cos_t + rot * sin_t


def build_triples(n, nouns):
    out = []
    for i in range(len(PERSONS)):
        for j in range(len(nouns)):
            if len(out) >= n:
                return out
            out.append((PERSONS[i], PERSONS[(i + 3) % len(PERSONS)], nouns[j]))
    return out


def span_of(tok, text, w, base):
    e = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    c0 = text.index(w, base) if base else text.index(w)
    sel = [i for i, (x, y) in enumerate(e['offset_mapping']) if x < c0 + len(w) and y > c0]
    return sel, e['input_ids']


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
    model.model.rotary_emb.register_forward_hook(lambda mod, i, o: cap.__setitem__('rope', o))

    # ---------- 阶段 1：身份方向 û（与目标句同构、同位置） ----------
    plen = {p: len(tok(p, add_special_tokens=False)['input_ids']) for p in PERSONS}
    from collections import Counter as _C
    pool = [p for p in PERSONS if plen[p] == _C(plen.values()).most_common(1)[0][0]]
    pairs = []
    while len(pairs) < N_CTX:
        a, b = rng.integers(0, len(pool), 2)
        if a != b:
            pairs.append((pool[a], pool[b]))

    nouns = NOUNS[:8] if SMOKE else NOUNS
    U, G, pos_of, kept_by_form = {}, {}, {}, {}
    for form in FORMS:
        tmpl = TMPL_UID[form]
        kept = []
        means = {}
        for w in nouns:
            ks, pset, ok = [], set(), True
            for A, B in pairs:
                text = tmpl.format(A=A, B=B, W=w)
                full = render(tok, text)
                base = full.find(text)
                sel, ids = span_of(tok, full, w, base)
                if len(sel) != 1:
                    ok = False
                    break
                pset.add(sel[0])
                with torch.no_grad():
                    cap.clear()
                    model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
                ks.append(np.stack([cap[li]['k'][0, sel[0]].float().cpu().numpy()
                                    for li in range(L)]))
            if not ok or len(pset) != 1:
                continue
            kept.append(w)
            mu = np.mean(np.stack(ks), axis=0)                      # [L,KV,HD]
            means[w] = (mu, np.stack(ks))
            pos_of[(form, w)] = pset.pop()
        # 三步分解：共同方向 g（所有词共有） → 身份方向 û（去共同后各词特有） → 上下文
        g = np.mean([means[w][0] for w in kept], axis=0)             # [L,KV,HD]
        G[form] = g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-9)
        for w in kept:
            mu, kk = means[w]
            perp = mu - (mu * G[form]).sum(-1, keepdims=True) * G[form]
            U[(form, w)] = perp / np.maximum(np.linalg.norm(perp, axis=-1, keepdims=True), 1e-9)
        kept_by_form[form] = kept
        print('%s: û 覆盖 %d/%d 词，theme 位置=%s'
              % (form, len(kept), len(nouns), sorted({pos_of[(form, w)] for w in kept})))
    print('阶段1 %.0fs' % (time.time() - t0))

    # ---------- 阶段 2：144 格的 s_id / s_ctx ----------
    trips = [t for t in build_triples(6 if SMOKE else 24, nouns)
             if ('A', t[2]) in U and ('C', t[2]) in U]
    sid, sctx, scmn, meta = [], [], [], []
    beh = {q: [0, 0] for q in QUERIES}

    for trip in trips:
        for form in FORMS:
            ctx, ranges = build(trip, form)
            for qk, qtext in QUERIES.items():
                user = ctx + qtext
                full = render(tok, user)
                base = full.find(user)
                c0, c1 = ranges['theme']
                c0, c1 = c0 + base, c1 + base
                e = tok(full, return_offsets_mapping=True, add_special_tokens=False)
                ids, offs = e['input_ids'], e['offset_mapping']
                tsel = [i for i, (a, b) in enumerate(offs) if a < c1 and b > c0]
                if len(tsel) != 1:
                    continue
                tpos = tsel[0]
                with torch.no_grad():
                    cap.clear()
                    out = model(input_ids=torch.tensor([ids], device=dev), use_cache=False)

                lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
                sc = {x: float(lp[tok(x, add_special_tokens=False)['input_ids'][0]]) for x in trip}
                want = {'ag': trip[0], 're': trip[1], 'th': trip[2]}[qk]
                beh[qk][1] += 1
                beh[qk][0] += int(max(sc, key=sc.get) == want)

                cos, sin = cap['rope']
                ct, st = cos[0, tpos].float(), sin[0, tpos].float()
                cl, sl = cos[0, -1].float(), sin[0, -1].float()
                u0 = torch.tensor(U[(form, trip[2])], device=dev).float()     # [L,KV,HD]
                ur = rope_vec(u0, ct, st)
                g0 = torch.tensor(G[form], device=dev).float()
                gr = rope_vec(g0, ct, st)

                a_id = np.zeros((L, QH), dtype=np.float32)
                a_ctx = np.zeros((L, QH), dtype=np.float32)
                a_cmn = np.zeros((L, QH), dtype=np.float32)
                for li in range(L):
                    q = cap[li]['q'][0, -1].float()                            # [QH,HD]
                    k = cap[li]['k'][0, tpos].float()                          # [KV,HD]
                    qr = rope_vec(q, cl, sl)
                    kr = rope_vec(k, ct, st)
                    gh, uh = gr[li], ur[li]
                    kcmn = (kr * gh).sum(-1, keepdim=True) * gh                 # 共同分量
                    perp = kr - kcmn
                    kid = (perp * uh).sum(-1, keepdim=True) * uh               # 身份分量
                    kct = perp - kid                                           # 上下文残差
                    f = lambda v: (qr * v.repeat_interleave(rep, dim=0)).sum(-1).cpu().numpy() * scale
                    a_cmn[li], a_id[li], a_ctx[li] = f(kcmn), f(kid), f(kct)
                sid.append(a_id)
                sctx.append(a_ctx)
                scmn.append(a_cmn)
                meta.append({'form': form, 'qk': qk, 'theme': trip[2]})
        if (len(sid) // 6) % 6 == 0:
            print('  三元组进度 %d/%d  %.0fs' % (len(sid) // 6, len(trips), time.time() - t0))

    S_id = np.stack(sid)          # [N, L, QH]
    S_ctx = np.stack(sctx)
    S_cmn = np.stack(scmn)
    S_act = S_id + S_ctx + S_cmn
    N = len(sid)
    print('有效格 %d，行为门 %s' % (N, {q: '%d/%d' % tuple(beh[q]) for q in QUERIES}))

    # ---------- 阶段 3：方差分解 ----------
    def shares(mask=None):
        P = {'cmn': S_cmn, 'id': S_id, 'ctx': S_ctx}
        if mask is not None:
            P = {k: v[mask] for k, v in P.items()}
        act = P['cmn'] + P['id'] + P['ctx']
        vab = act.var(axis=0)                                   # [L,QH]
        out = {k: v.var(axis=0) / np.maximum(vab, 1e-12) for k, v in P.items()}
        out['sum'] = sum(out[k] for k in P)
        rho = np.array([[float(np.corrcoef(P['id'][:, l, h], act[:, l, h])[0, 1])
                         for h in range(QH)] for l in range(L)])
        return out, act, rho

    SH, S_act2, rho = shares()
    sh_id, sh_sum = SH['id'], SH['sum']
    prof = np.median(sh_id, axis=1)                 # [L]
    n3 = L // 3
    front, mid, back = (float(prof[:n3].mean()), float(prof[n3:2 * n3].mean()),
                        float(prof[2 * n3:].mean()))
    j1 = (front > 0.5 and back < 0.2) or (front < 0.2 and back > 0.5)
    bimodal = []
    for l in range(L):
        v = sh_id[l]
        bimodal.append(float(abs(v.mean() - np.median(v)) / max(v.std(), 1e-9)))
    j2 = float(np.median(bimodal)) > 0.3
    j3 = 0.35 <= float(np.median(prof)) <= 0.65 and not j1 and not j2
    verdict = 'J1_boundary' if j1 else ('J2_split' if j2 else ('J3_equal' if j3 else 'NULL'))

    print('\n判决=%s' % verdict)
    print('share_id  剖面：' + ' '.join('%.2f' % x for x in prof))
    print('share_cmn 剖面：' + ' '.join('%.2f' % x for x in np.median(SH['cmn'], axis=1)))
    print('share_ctx 剖面：' + ' '.join('%.2f' % x for x in np.median(SH['ctx'], axis=1)))
    print('前/中/后 = %.3f / %.3f / %.3f   share 和 中位=%.3f（偏离 1 说明两项相关）'
          % (front, mid, back, float(np.median(sh_sum))))
    print('corr(s_id,s_act) 中位=%.3f' % float(np.median(rho)))

    by_qk = {}
    for qk in QUERIES:
        m = np.array([x['qk'] == qk for x in meta])
        if m.sum() > 5:
            o, _, _ = shares(m)
            by_qk[qk] = [round(float(x), 4) for x in np.median(o['id'], axis=1)]

    rec = {
        'seed': SEED, 'smoke': SMOKE, 'n_cells': N, 'n_trials': len(trips),
        'behaviour_gate': {q: '%d/%d' % tuple(beh[q]) for q in QUERIES},
        'verdict': verdict,
        'share_id_profile': [round(float(x), 4) for x in prof],
        'front_mid_back': [round(front, 4), round(mid, 4), round(back, 4)],
        'share_sum_median': round(float(np.median(sh_sum)), 4),
        'share_cmn_profile': [round(float(x), 4) for x in np.median(SH['cmn'], axis=1)],
        'share_ctx_profile': [round(float(x), 4) for x in np.median(SH['ctx'], axis=1)],
        'share_cmn_median': round(float(np.median(SH['cmn'])), 4),
        'share_ctx_median': round(float(np.median(SH['ctx'])), 4),
        'bimodality_median': round(float(np.median(bimodal)), 4),
        'share_id_by_query': by_qk,
        'share_id_head_matrix': np.round(sh_id, 4).tolist(),
        'predicted_boundary': 'L20-L33',
        'elapsed_s': round(time.time() - t0, 1),
    }
    provenance.write_result('pat_q3t_t2', rec, model_path=MODEL_PATH, seed=SEED, tool='pat_q3t_t2.py')
    print('完成 %.0fs' % (time.time() - t0))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
