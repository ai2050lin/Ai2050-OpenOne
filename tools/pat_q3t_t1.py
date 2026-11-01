# -*- coding: utf-8 -*-
"""Q3-T1：齿痕剖面 —— 同一 token 在不同上下文里的 key 是否共享一个「身份」方向。

预注册：research/gpt5/docs/PAT_Q3T_PREREG_v1.md（§1）

两处勘误（都在正式跑之前记下，不是跑完再改）
--------------------------------------------
勘误 1（结构）：预注册写"36 层 × 32 头"对 key 侧是错的。Qwen3-4B 是 GQA，
    32 个 query 头只共享 **8 套 K/V**，齿痕只在 8 个 kv 头上定义。
勘误 2（判据，SMOKE 暴露）：预注册的主判据是 R_id = ‖mean k‖ / mean ‖k‖。
    实测 R_id 全层 0.995–1.000，四选一形状里只剩"平坦"——判据没有分辨力。
    原因：R_id 只回答"这些 key 有多共线"，不回答"那条共同线是不是**身份**"。
    任意两个不同名词的 key 也可能高度共线，此时 R_id 一样高。
    ⇒ 主判据换成 **身份检索率 match(l, kv)**：用该层该头的 key 与全部词的
      身份原型方向比余弦，top-1 命中自己即正确。随机基线 1/|W|。
      并列报 sep = mean_cos(同词) − mean_cos(异词) 作为效应量。
    R_id 降级为描述量，不再作判据。

设计：位置严格对齐是这一测的全部
--------------------------------
key 是 RoPE 之后的量，位置变了方向就转，跨位置比较得到的是纯伪信号。所以：

  模板 P9  「{A}送给了{B}一个{W}。」   W 起始字符 9（主动句 theme 槽）
  模板 P0  「{W}被{A}送给了{B}。」     W 起始字符 0（被动句 theme 槽）

两个模板各自内部位置恒定（探针已验：同一模板下 W 的 token 索引唯一）。
A、B 恒取 2 字人名，所以主题槽索引不随实体改变。
每个 (模板, 词) 取 M 个不同 (A,B) 上下文，唯一变量是周围词。

判据（跑前冻结，作用于 match 的层剖面，两模板各算一次）
------------------------------------------------------
  U 形     前1/3均值 − 中1/3均值 ≥ 0.10 且 后1/3均值 − 中1/3均值 ≥ 0.10
  单调降   Spearman ρ(层, match) ≤ −0.7
  单调升   Spearman ρ(层, match) ≥ +0.7
  平坦     36 层极差 < 0.10
身份有效性闸：match 的 36 层均值 ≥ 5×随机基线，否则判"key 不含可分身份"，本轮 NULL。

预注册预测：U 形。

用法
----
  SMOKE=1 .venv/Scripts/python.exe -X utf8 tools/pat_q3t_t1.py
  .venv/Scripts/python.exe -X utf8 tools/pat_q3t_t1.py
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
from pat_q1_unbind import PERSONS, SYS

MODEL_PATH = 'D:/develop/model/Qwen/Qwen3-4B'
SEED = 20261101
SMOKE = os.environ.get('SMOKE', '') == '1'
N_CTX = 5 if SMOKE else 20

NOUNS = ['苹果', '香蕉', '橘子', '桃子', '面包', '蛋糕', '杯子', '碗', '毛巾', '雨伞',
         '铅笔', '书包', '钥匙', '手表', '眼镜', '帽子', '围巾', '手套', '鞋子', '镜子',
         '想法', '计划', '建议', '问题', '答案', '理由', '办法', '习惯', '故事',
         '照片', '地图', '礼物', '玩具', '花瓶', '台灯', '篮球', '吉他', '茶杯', '枕头']

TMPLS = {'P9': '{A}送给了{B}一个{W}。', 'P0': '{W}被{A}送给了{B}。'}


def render(tok, user_text):
    return tok.apply_chat_template(
        [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': user_text}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False)


def w_span(tok, text, w):
    full = render(tok, text)
    base = full.find(text)
    enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
    c0 = text.index(w) + base
    sel = [i for i, (a, b) in enumerate(enc['offset_mapping']) if a < c0 + len(w) and b > c0]
    if len(sel) != 1:
        raise ValueError('not single token')
    return full, enc['input_ids'], sel[0]


def ctx_pairs(pool, n, rng):
    pairs, seen = [], set()
    guard = 0
    while len(pairs) < n and guard < 10000:
        guard += 1
        a, b = rng.integers(0, len(pool), 2)
        if a == b or (a, b) in seen:
            continue
        seen.add((a, b))
        pairs.append((pool[a], pool[b]))
    return pairs


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d else 0.0


def shape_of(prof):
    n = len(prof)
    a, m, b = prof[: n // 3].mean(), prof[n // 3: 2 * n // 3].mean(), prof[2 * n // 3:].mean()
    rho = spearman(np.arange(n), prof)
    if a - m >= 0.10 and b - m >= 0.10:
        return 'U', rho
    if rho <= -0.7:
        return 'monotone_down', rho
    if rho >= 0.7:
        return 'monotone_up', rho
    if prof.max() - prof.min() < 0.10:
        return 'flat', rho
    return 'other', rho


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    dev = 'cuda'

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, attn_implementation='eager').to(dev).eval()
    cfg = model.config
    L, KV = cfg.num_hidden_layers, cfg.num_key_value_heads
    HD = getattr(cfg, 'head_dim', 128)

    cand = NOUNS[:8] if SMOKE else NOUNS
    # 位置恒定要求：人名的 token 长度必须一致（"张三"可能 1 token 也可能 2 token）。
    # 取 token 长度最多数的那一组人名作为上下文池，位置才不随 A/B 变化。
    plen = {p: len(tok(p, add_special_tokens=False)['input_ids']) for p in PERSONS}
    from collections import Counter as _C
    mode_len = _C(plen.values()).most_common(1)[0][0]
    pool = [p for p in PERSONS if plen[p] == mode_len]
    print('人名 token 长度=%s，取 %d-token 池 %d 个：%s'
          % (sorted(set(plen.values())), mode_len, len(pool), pool))
    pairs = ctx_pairs(pool, N_CTX, rng)

    jobs, kept = [], {}
    for pos_name, tmpl in TMPLS.items():
        good = []
        for w in cand:
            try:
                spans = [w_span(tok, tmpl.format(A=A, B=B, W=w), w) for A, B in pairs]
            except ValueError:
                continue
            if len({s[2] for s in spans}) != 1:
                continue
            good.append(w)
            for ci, (full, ids, pidx) in enumerate(spans):
                jobs.append((pos_name, w, ci, ids, pidx))
        kept[pos_name] = good
        print('%s 可用词 %d/%d，位置=%s' % (
            pos_name, len(good), len(cand), sorted({j[4] for j in jobs if j[0] == pos_name})))
        if len(good) < (4 if SMOKE else 8):
            raise RuntimeError('%s 可用词太少' % pos_name)

    N = len(jobs)
    K = np.zeros((N, L, KV, HD), dtype=np.float32)
    cap = {}

    def mk(li):
        def hook(mod, inp, out):
            cap[li] = out.detach()
        return hook

    for li in range(L):
        model.model.layers[li].self_attn.k_norm.register_forward_hook(mk(li))

    print('样本 %d 条，开始采集…' % N)
    with torch.no_grad():
        for i, (pos_name, w, ci, ids, pidx) in enumerate(jobs):
            cap.clear()
            model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
            for li in range(L):
                K[i, li] = cap[li][0, pidx].float().cpu().numpy()
            if (i + 1) % 300 == 0:
                print('  %d/%d  %.0fs' % (i + 1, N, time.time() - t0))

    pos_arr = np.array([j[0] for j in jobs])
    w_arr = np.array([j[1] for j in jobs])

    summary, per_head = {}, {}
    for pos_name in TMPLS:
        nouns = kept[pos_name]
        sel = np.where(pos_arr == pos_name)[0]
        Kp, wp = K[sel], w_arr[sel]
        n, W = len(sel), len(nouns)

        Kn = Kp / np.maximum(np.linalg.norm(Kp, axis=-1, keepdims=True), 1e-9)
        same = (wp[:, None] == wp[None, :]) & ~np.eye(n, dtype=bool)
        diff = ~(wp[:, None] == wp[None, :])

        match = np.zeros((L, KV))
        cos_same = np.zeros((L, KV))
        cos_diff = np.zeros((L, KV))
        protos = {}
        for li in range(L):
            c = np.einsum('nkd,mkd->nmk', Kn[:, li], Kn[:, li])
            cos_same[li] = c[same].mean(axis=0)
            cos_diff[li] = c[diff].mean(axis=0)
            for k in range(KV):
                for w in nouns:
                    v = Kn[np.where(wp == w)[0], li, k].mean(axis=0)
                    protos[w] = v / max(np.linalg.norm(v), 1e-9)
                P = np.stack([protos[x] for x in nouns])        # [W, HD]
                hits = []
                for wi, w in enumerate(nouns):
                    idx = np.where(wp == w)[0]
                    pred = np.argmax(Kn[idx, li, k] @ P.T, axis=1)
                    hits.append((pred == wi).mean())
                match[li, k] = float(np.mean(hits))

        sep = cos_same - cos_diff
        m_prof = np.median(match, axis=1)
        s_prof = np.median(sep, axis=1)
        shape, rho = shape_of(m_prof)
        base = 1.0 / W
        summary[pos_name] = {
            'nouns': nouns, 'n_seq': int(n), 'chance': round(base, 4),
            'match_profile': [round(float(x), 4) for x in m_prof],
            'match_mean': round(float(m_prof.mean()), 4),
            'match_first_half': round(float(m_prof[: L // 2].mean()), 4),
            'match_second_half': round(float(m_prof[L // 2:].mean()), 4),
            'sep_profile': [round(float(x), 4) for x in s_prof],
            'sep_median': round(float(np.median(s_prof)), 4),
            'cos_same_L0': round(float(np.median(cos_same[0])), 4),
            'cos_diff_L0': round(float(np.median(cos_diff[0])), 4),
            'cos_same_L35': round(float(np.median(cos_same[-1])), 4),
            'cos_diff_L35': round(float(np.median(cos_diff[-1])), 4),
            'shape': shape, 'spearman': round(rho, 3),
            'peak_layer': int(np.argmax(m_prof)), 'trough_layer': int(np.argmin(m_prof)),
            'identity_gate': bool(m_prof.mean() >= 5 * base),
            'fold_over_chance': round(float(m_prof.mean() / base), 2),
        }
        per_head[pos_name] = {'match': np.round(match, 4).tolist(),
                              'sep': np.round(sep, 4).tolist()}
        print('\n[%s] 形状=%s rho=%.3f 峰=L%d 谷=L%d  检索率均值=%.4f（随机 %.4f，%.1f×）'
              % (pos_name, shape, rho, summary[pos_name]['peak_layer'],
                 summary[pos_name]['trough_layer'], m_prof.mean(), base, m_prof.mean() / base))
        print('    cos 同词 L0/L35=%.4f/%.4f   cos 异词 L0/L35=%.4f/%.4f   sep 中位=%.4f'
              % (summary[pos_name]['cos_same_L0'], summary[pos_name]['cos_same_L35'],
                 summary[pos_name]['cos_diff_L0'], summary[pos_name]['cos_diff_L35'],
                 summary[pos_name]['sep_median']))
        print('    检索剖面：' + ' '.join('%.2f' % x for x in m_prof))
        print('    sep 剖面：' + ' '.join('%.2f' % x for x in s_prof))

    rec = {'seed': SEED, 'smoke': SMOKE, 'n_ctx': N_CTX, 'n_seq': N,
           'summary': summary, 'per_head': per_head,
           'elapsed_s': round(time.time() - t0, 1)}
    provenance.write_result('pat_q3t_t1', rec, model_path=MODEL_PATH, seed=SEED, tool='pat_q3t_t1.py')
    print('\n完成 %.1fs' % (time.time() - t0))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
