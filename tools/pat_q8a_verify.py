# -*- coding: utf-8 -*-
"""Q8-A 独立复核：只从磁盘（模型权重 + 结果 json）重算，不读任何结论字段。

口径刻意与主脚本不同：
  - 余弦用 np.dot / 手写归一，不用主脚本的 cosine()
  - 均值用 float64 累加
  - 2AFC 用欧氏距离直接算，不用 1-cos
每个检查都给 (重算值, json 值, 差, 阈值, PASS/FAIL)。任一项 FAIL ⇒ 整轮不入账。

用法：.venv/Scripts/python.exe -X utf8 tools/pat_q8a_verify.py <result.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-4
CHECKS = []


def chk(name, got, want, tol=TOL, note=''):
    ok = (got is not None and want is not None
          and np.isfinite(got) and np.isfinite(want) and abs(got - want) <= tol)
    CHECKS.append(dict(name=name, got=float(got) if got is not None else None,
                       want=float(want) if want is not None else None,
                       diff=float(got - want) if (got is not None and want is not None) else None,
                       tol=tol, pass_=bool(ok), note=note))
    print('%-46s got=%+.6f want=%+.6f diff=%+.2e %s %s'
          % (name, got, want, got - want, 'PASS' if ok else 'FAIL', note), flush=True)
    return ok


def main():
    rp = Path(sys.argv[1] if len(sys.argv) > 1 else
              'results/2026-11-01/091106_pat_q8a.json')
    doc = json.loads(rp.read_text(encoding='utf-8'))
    P = doc['payload']
    mp = doc['provenance']['model_path']
    tok = AutoTokenizer.from_pretrained(mp)
    model = AutoModelForCausalLM.from_pretrained(mp, dtype=torch.bfloat16)
    model.eval()
    E = model.get_input_embeddings().weight.detach().to(torch.float64).cpu().numpy()
    mu = E.sum(0) / E.shape[0]
    dec = [tok.decode([i]) for i in range(E.shape[0])]
    print('model', mp, E.shape, flush=True)

    def unit(X):
        n = np.sqrt((X * X).sum(-1, keepdims=True))
        return X / np.where(n == 0, 1, n)

    def wcos(ids):
        X = unit(E[ids] - mu)
        S = X @ X.T
        n = len(ids)
        return float((S.sum() - np.trace(S)) / (n * (n - 1)))

    def within_pooled(ids_list):
        X = unit(np.array([E[i].mean(0) for i in ids_list]) - mu)
        S = X @ X.T
        n = len(ids_list)
        return float((S.sum() - np.trace(S)) / (n * (n - 1)))

    # 1 单位分层：fragment vs word 的种群内平均余弦
    frag = [i for i, s in enumerate(dec) if '\ufffd' in s]
    word = [i for i, s in enumerate(dec) if len(s) >= 2 and all(
        '\u4e00' <= c <= '\u9fff' for c in s)]
    rng = np.random.default_rng(7)
    f_s = rng.choice(frag, min(1200, len(frag)), replace=False)
    w_s = rng.choice(word, min(1200, len(word)), replace=False)
    chk('P1 fragment within', wcos(f_s), P['P1']['fragment']['within_cos_centered'], 5e-3,
        '抽样不同，容差放宽')
    chk('P1 word within', wcos(w_s), P['P1']['word']['within_cos_centered'], 5e-3, '抽样不同')

    # 2 频率—范数
    from wordfreq import zipf_frequency as zf
    def cjk1(s):
        return len(s) == 1 and ('\u3400' <= s <= '\u9fff' or '\uf900' <= s <= '\ufaff')

    chars = [i for i, s in enumerate(dec) if cjk1(s) and zf(s, 'zh') > 0]
    nn = np.sqrt((E[chars] ** 2).sum(1))
    zz = np.array([zf(dec[i], 'zh') for i in chars])
    from scipy.stats import spearmanr
    chk('P2 ρ(log‖e‖, zipf)', float(spearmanr(np.log(nn), zz).statistic),
        P['P2']['rho'], 1e-5, 'n=%d' % len(chars))

    # 3 类别：within(FRUIT) / within(COLOR)
    def pool(name):
        return [tok.encode(w, add_special_tokens=False) for w in P['sets'][name]['words']]

    chk('P3 within(FRUIT)', within_pooled(pool('FRUIT')), P['P3']['within_fruit'])
    chk('P3 within(COLOR)', within_pooled(pool('COLOR')), P['P3']['within_color'])

    # 3b 2AFC 用欧氏距离重算（不同口径）
    def afc(A_ids, B_ids):
        A = np.array([E[i].mean(0) for i in A_ids]) - mu
        B = np.array([E[i].mean(0) for i in B_ids]) - mu
        ok = 0
        for i in range(len(A)):
            ca = A[np.arange(len(A)) != i].mean(0)
            cb = B.mean(0)
            ok += int(np.linalg.norm(A[i] - ca) < np.linalg.norm(A[i] - cb))
        return ok / len(A)

    # 口径不同（欧氏 vs 1−cos），容许 0.06 的差异；方向必须一致
    chk('P3 2AFC FRUIT→COLOR(欧氏)', afc(pool('FRUIT'), pool('COLOR')),
        P['P3']['afc'][0]['acc'], 0.06, '欧氏 vs 1−cos，口径差')

    # 4 色相 Mantel 的输入侧：重算色相角的圆周距离与 cos 距离的 Spearman
    lab = P['P4']['lab']
    ws = P['P4']['words']
    h = np.array([lab[w]['h'] for w in ws])
    D = np.abs(h[:, None] - h[None, :])
    D = np.minimum(D, 360 - D)
    ids = [tok.encode(w, add_special_tokens=False) for w in ws]
    V = unit(np.array([E[i].mean(0) for i in ids]) - mu)
    Dc = 1 - V @ V.T
    iu = np.triu_indices(len(ws), 1)
    chk('P4 Mantel ρ(色相,cos)', float(spearmanr(D[iu], Dc[iu]).statistic),
        P['P4']['hue']['rho'], 1e-6)

    # 5 覆盖：全 CJK 普查
    cnt = 0
    for cp in range(0x4E00, 0x9FA5 + 1):
        if len(tok.encode(chr(cp), add_special_tokens=False)) > 1:
            cnt += 1
    chk('P5 覆盖（字节回退字数）', float(cnt), float(P['P5']['census_byte_fallback']), 0.5)

    # 5b within(RARE) / within(COMMON)
    chk('P5 within(RARE)', within_pooled(pool('RARE')), P['P5']['collapse']['within_rare'])
    com = [tok.encode(w, add_special_tokens=False)[0] for w in P['sets']['COMMON']['words']]
    chk('P5 within(COMMON)', wcos(com), P['P5']['collapse']['within_common'])

    # 6 形旁：石组 Δ（用 json 里的 between 重算）
    rad = [tok.encode(c, add_special_tokens=False)[0] for c in '硬碰碗破碎砖砂码磁确']
    X = unit(E[rad] - mu)
    S = X @ X.T
    n = len(rad)
    w = float((S.sum() - np.trace(S)) / (n * (n - 1)))
    chk('P6 石 within', w, P['P6']['groups']['石']['within'])
    chk('P6 石 Δ(=within−between)', w - P['P6']['between'], P['P6']['groups']['石']['diff'])

    # 7 频率轴单坐标
    allw = []
    for nm in ('COLOR', 'FRUIT', 'RARE', 'COMMON'):
        allw += P['sets'][nm]['words']
    ids = [tok.encode(w, add_special_tokens=False) for w in allw]
    Xw = np.array([E[i].mean(0) for i in ids])
    zw = np.array([zf(w, 'zh') for w in allw])
    zc = (zw - zw.mean()) / zw.std()
    Xc = (Xw - Xw.mean(0)) / Xw.std(0)
    rf = (Xc * zc[:, None]).mean(0)
    chk('P7 单坐标|ρ(坐标,zipf)|', float(np.abs(rf).max()),
        abs(P['P7']['freq_axis']['rho_max']))

    # 9 中英对译秩：抽 10 对重算
    G = unit(E - mu)
    pairs = P['P9']['rows'][:10]
    ranks = []
    for r in pairs:
        i = tok.encode(r['zh'], add_special_tokens=False)[0]
        v = G[i]
        s = G @ v
        s[i] = -9
        t = tok.encode(r['tok'], add_special_tokens=False)
        if len(t) != 1:
            continue
        ranks.append(int((s > s[t[0]]).sum()) + 1)
    chk('P9 前10对名次中位', float(np.median(ranks)),
        float(np.median([r['rank'] for r in pairs[:len(ranks)]])), 0.5)

    # 10 简繁 cos 中位
    rows = P['P10']['rows']
    cs = []
    for r in rows:
        a = tok.encode(r['simp'], add_special_tokens=False)
        b = tok.encode(r['trad'], add_special_tokens=False)
        if len(a) != 1 or len(b) != 1:
            continue
        cs.append(float(G[a[0]] @ G[b[0]]))
    chk('P10 简繁 cos 中位', float(np.median(cs)), P['P10']['cos_median'])

    npass = sum(c['pass_'] for c in CHECKS)
    print('\n%d/%d PASS' % (npass, len(CHECKS)), flush=True)
    out = rp.with_name(rp.stem + '_verify.json')
    out.write_text(json.dumps(dict(result=str(rp), n_pass=int(npass), n=len(CHECKS),
                                   checks=CHECKS), ensure_ascii=False, indent=1),
                   encoding='utf-8', newline='')
    print('VERIFY', out)
    return 0 if npass == len(CHECKS) else 1


if __name__ == '__main__':
    sys.exit(main())
