# -*- coding: utf-8 -*-
"""Q7-A 独立复核：只从磁盘上的 result.json 重算，不读任何结论字段。

复核项（每项独立打印 PASS/FAIL）：
  V1  文件内容 sha8 复算
  V2  设计完整性（162 句 / 54-54-54 / 18 内容各 9 句 / 句子互不相同）
  V3  刺激不变式（同内容同表面，三个关系的前缀逐字相同到 A 后标点）
  V4  J0 对齐自检：P_AE 每表面准确率必须恰为 1/3
  V5  J1 准确率复算 + 内容簇 bootstrap CI 复算（容差 0.03）
  V6  分关系 / 分表面准确率复算
  V7  J2 迁移准确率复算（含 hit 掩码）
  V8  J3 秩曲线复算
  V9  J4/J5/J6 基线复算
  V10 McNemar 不一致对与 p 值复算
  V11 零分布边界：J1 是否高于 null_p95

用法：.venv/Scripts/python.exe -X utf8 tools/pat_q7a_verify.py [结果文件]
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
FLOOR = 1.0 / 3.0
SEED = 20261101
TOL = 1e-4   # 结果文件准确率保留 4 位小数


def load(path=None):
    if path is None:
        cand = sorted(RESULTS.rglob('*_pat_q7a.json'))
        if not cand:
            raise SystemExit('找不到 pat_q7a 结果文件')
        path = cand[-1]
    p = Path(path)
    doc = json.loads(p.read_text(encoding='utf-8'))
    return p, doc


def content_sha8(doc):
    d = dict(doc)
    d.pop('sha8', None)
    body = json.dumps(d, ensure_ascii=False, indent=1)
    return hashlib.sha256(body.encode('utf-8')).hexdigest()[:8]


def main() -> int:
    path, doc = load(sys.argv[1] if len(sys.argv) > 1 else None)
    pl = doc['payload']
    ex = pl['pred_export']
    y = np.array(ex['labels'])
    ci = np.array(ex['content'])
    sf = np.array(ex['surf'])
    n = len(y)
    texts = pl['texts']
    res = []

    def chk(tag, ok, msg=''):
        res.append((tag, bool(ok)))
        print('%s %-6s %s' % ('PASS' if ok else 'FAIL', tag, msg))

    # V1
    got = content_sha8(doc)
    chk('V1', got == doc['sha8'], 'sha8 复算 %s vs 文件 %s' % (got, doc['sha8']))

    # V2
    cnt = np.bincount(y, minlength=3).tolist()
    per_c = [int((ci == g).sum()) for g in range(18)]
    ok2 = (n == 162 and cnt == [54, 54, 54]
           and sorted(set(per_c)) == [9] and len(set(texts)) == n)
    chk('V2', ok2, 'n=%d 类别=%s 每内容句数=%s 唯一句=%d' % (n, cnt, sorted(set(per_c)), len(set(texts))))

    # V3 前缀不变式：同内容同表面，截到 A 后第一个标点必须完全相同
    def lcp(strs):
        if not strs:
            return ''
        a = min(strs)
        b = max(strs)
        k = 0
        while k < len(a) and k < len(b) and a[k] == b[k]:
            k += 1
        return a[:k]

    ok3 = True
    detail = ''
    for c in range(18):
        idx9 = [i for i in range(n) if ci[i] == c]
        A = lcp([texts[i] for i in idx9])          # 9 句的公共前缀就是 A 本身
        if len(A) < 8:
            ok3 = False
            detail = '内容%d 的 A 过短（%d 字）' % (c, len(A))
            break
        for s in ('S1', 'S2', 'S3'):
            idx = [i for i in range(n) if ci[i] == c and sf[i] == s]
            lp = lcp([texts[i] for i in idx])
            # 三个关系的 B 可以共享开头词（如都以「今天上午」起），所以只要求
            # 公共前缀至少覆盖 A + 紧随 A 的标点，而不是恰好停在标点处。
            if not (len(lp) >= len(A) + 1 and lp[len(A)] in '，。'):
                ok3 = False
                detail = '内容%d %s 公共前缀未覆盖 A+标点（lp=%d, A=%d）' % (
                    c, s, len(lp), len(A))
                break
        if not ok3:
            break
    chk('V3', ok3, detail or '18 内容 × 3 表面：公共前缀覆盖 A + 紧随标点，A 在三个关系间逐字相同')

    # V4 J0
    j0 = {}
    for s in ('S1', 'S2', 'S3'):
        m = sf == s
        j0[s] = float((np.array(ex['P_AE'])[m] == y[m]).mean())
    dev = max(abs(v - FLOOR) for v in j0.values())
    chk('V4', dev < 1e-6, 'P_AE 各表面 %s 最大偏离 %.2e'
        % ({k: round(v, 6) for k, v in j0.items()}, dev))

    # V5 J1 + CI
    pred = np.array(ex['P_END'])
    acc = float((pred == y).mean())
    chk('V5a', abs(acc - pl['J1_acc']) < TOL, 'J1 复算 %.4f vs 报告 %.4f' % (acc, pl['J1_acc']))
    rng = np.random.default_rng(SEED + 101)
    corr = (pred == y).astype(float)
    uniq = np.unique(ci)
    vals = []
    for _ in range(4000):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        m = np.concatenate([np.where(ci == g)[0] for g in pick])
        vals.append(corr[m].mean())
    lo, hi = float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
    rlo, rhi = pl['J1_ci']
    ok5 = (abs(lo - rlo) <= 0.03 and abs(hi - rhi) <= 0.03
           and rlo - TOL <= acc <= rhi + TOL)
    chk('V5b', ok5, 'CI 复算 [%.4f, %.4f] vs 报告 [%.4f, %.4f]；点估计在 CI 内' % (lo, hi, rlo, rhi))

    # V6
    per_rel = {}
    for k in range(3):
        m = y == k
        per_rel[k] = round(float((pred[m] == y[m]).mean()), 4)
    ok6a = all(abs(per_rel[k] - pl['per_rel'][kk]) < TOL
               for k, kk in zip(range(3), ['seq', 'con', 'cau']))
    per_surf = {}
    for s in ('S1', 'S2', 'S3'):
        m = sf == s
        per_surf[s] = round(float((pred[m] == y[m]).mean()), 4)
    ok6b = all(abs(per_surf[s] - pl['per_surf'][s]) < TOL for s in ('S1', 'S2', 'S3'))
    chk('V6', ok6a and ok6b, '分关系 %s 分表面 %s' % (per_rel, per_surf))

    # V7 J2
    j2p = np.array(ex['J2'])
    hit = np.array(ex['J2_hit']).astype(bool)
    j2 = float((j2p[hit] == y[hit]).mean())
    chk('V7', abs(j2 - pl['J2_main']) < TOL,
        'J2 复算 %.4f vs 报告 %.4f（hit=%d）' % (j2, pl['J2_main'], int(hit.sum())))

    # V8 J3
    j3 = {}
    for k in (1, 2, 3):
        pk = np.array(ex['J3'][str(k)] if str(k) in ex['J3'] else ex['J3'][k])
        j3[k] = round(float((pk == y).mean()), 4)
    ok8 = all(abs(j3[k] - pl['J3'][str(k)]) < TOL for k in (1, 2, 3))
    chk('V8', ok8, 'J3 复算 %s vs 报告 %s' % (j3, pl['J3']))

    # V9
    a4 = float((np.array(ex['J4']) == y).mean())
    a5 = float((np.array(ex['J5']) == y).mean())
    a6 = float((np.array(ex['J6']) == y).mean())
    ok9 = (abs(a4 - pl['J4_char_bag']) < TOL and abs(a5 - pl['J5_length']) < TOL
           and abs(a6 - pl['J6_negation']) < TOL)
    chk('V9', ok9, 'J4=%.4f J5=%.4f J6=%.4f' % (a4, a5, a6))

    # V10 McNemar
    a_ok = pred == y
    b_ok = np.array(ex['J4']) == y
    nb = int(np.sum(a_ok & ~b_ok))
    nc = int(np.sum(~a_ok & b_ok))
    nn = nb + nc
    k = min(nb, nc)
    pmc = 1.0 if nn == 0 else min(1.0, 2.0 * sum(math.comb(nn, j) for j in range(k + 1)) / 2.0 ** nn)
    ok10 = (nb == pl['mcnemar']['j1_only'] and nc == pl['mcnemar']['j4_only']
            and abs(pmc - pl['mcnemar']['p']) < 1e-5)
    chk('V10', ok10, '不一致 %d vs %d，p 复算 %.6g vs 报告 %.6g' % (nb, nc, pmc, pl['mcnemar']['p']))

    # V11
    ok11 = acc > pl['null_p95'] and pl['null_p95'] > FLOOR - 0.02
    chk('V11', ok11, 'J1 %.4f > null_p95 %.4f；null_mean %.4f'
        % (acc, pl['null_p95'], pl['null_mean']))

    npass = sum(1 for _, ok in res if ok)
    print()
    print('复核 %d/%d 通过' % (npass, len(res)))
    print('文件 %s' % path)
    return 0 if npass == len(res) else 1


if __name__ == '__main__':
    raise SystemExit(main())
