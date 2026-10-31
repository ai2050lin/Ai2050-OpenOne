# -*- coding: utf-8 -*-
"""H6 结果分析：删单侧 vs 删双侧。默认按严格口径（排除解析失败题）出全部对照。

为什么必须严格口径：H5 的「P = rand、双向抵消」就是被解析失败污染出来的（见 R-MET-06）。
删 top-97 会顺手把「没写出答案行」的题修好，于是条件侧正确率被人为抬高、
base 侧被人为压低，7.3 点的真实差距被抹平。所以本脚本的主表只用
「两侧都给出有效答案」的题，原始口径只作对照展示。

用法：
  .venv/Scripts/python.exe -X utf8 tools/mech_pair_sign_analyze.py tempdata/pair_sign_L34.json
  .venv/Scripts/python.exe -X utf8 tools/mech_pair_sign_analyze.py tempdata/pair_sign_L34.json --json out.json
"""
import json
import sys
from math import comb
from pathlib import Path


def mcnemar(b, c):
    """两点式 McNemar 精确检验（双侧）。b/c: 只有一方为真的题数。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(0, k + 1)) / 2.0 ** n)


def paired(recs, ca, cb, gold_key='gold'):
    """两个条件的配对比较：正确率、只 A 对 / 只 B 对、McNemar p、改答数。"""
    ok_a = [r['out'][ca]['pred'] == r[gold_key] for r in recs]
    ok_b = [r['out'][cb]['pred'] == r[gold_key] for r in recs]
    only_a = sum(1 for x, y in zip(ok_a, ok_b) if x and not y)
    only_b = sum(1 for x, y in zip(ok_a, ok_b) if y and not x)
    chg = sum(1 for r in recs if r['out'][ca]['pred'] != r['out'][cb]['pred'])
    return {'acc_a': round(sum(ok_a) / len(recs), 3), 'acc_b': round(sum(ok_b) / len(recs), 3),
            'only_a_right': only_a, 'only_b_right': only_b,
            'p': round(mcnemar(only_a, only_b), 4), 'changed': chg, 'n': len(recs)}


def net_of(recs, cond, base='base'):
    """以 base 为基准的净方向：由错变对 − 由对变错。"""
    w = sum(1 for r in recs if r['out'][base]['pred'] == r['gold'] and r['out'][cond]['pred'] != r['gold'])
    rt = sum(1 for r in recs if r['out'][base]['pred'] != r['gold'] and r['out'][cond]['pred'] == r['gold'])
    return {'wrong': w, 'right': rt, 'net': rt - w}


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else 'tempdata/pair_sign_L34.json')
    out_json = None
    if '--json' in sys.argv:
        out_json = Path(sys.argv[sys.argv.index('--json') + 1])
    d = json.loads(src.read_text(encoding='utf-8'))
    recs = d['records']
    conds = [c for c in (d.get('summary', {}).get('conds') or []) if c != 'base']
    if not conds:
        conds = [c for c in recs[0]['out'] if c != 'base']
    print('文件 %s | n=%d | 条件 %s | 单元数 %s'
          % (src.name, len(recs), ['base'] + conds, d.get('summary', {}).get('n_units')))

    # ---- 口径对照：解析失败数 ----
    print('\n[1] 解析失败数（None）')
    for c in ['base'] + conds:
        print('    %-6s None %2d' % (c, sum(1 for r in recs if r['out'][c]['pred'] is None)))

    # ---- 主表：严格口径 ----
    sub = [r for r in recs if all(r['out'][c]['pred'] is not None for c in ['base'] + conds)]
    print('\n[2] 严格口径（所有条件都给出有效答案）：%d 题' % len(sub))
    base_acc = sum(1 for r in sub if r['out']['base']['pred'] == r['gold']) / len(sub)
    print('    base   %.3f' % base_acc)
    for c in conds:
        a = sum(1 for r in sub if r['out'][c]['pred'] == r['gold']) / len(sub)
        n_ = net_of(sub, c)
        print('    %-6s %.3f  (%+.1f 点)  净 %+3d（%d 错 / %d 对）'
              % (c, a, 100 * (a - base_acc), n_['net'], n_['wrong'], n_['right']))

    # ---- 配对检验 ----
    print('\n[3] 配对检验（严格口径）')
    pairs = [('base', c) for c in conds]
    if 'pos' in conds and 'neg' in conds:
        pairs.append(('pos', 'neg'))
    if 'pos' in conds and 'both' in conds:
        pairs.append(('pos', 'both'))
    if 'neg' in conds and 'both' in conds:
        pairs.append(('neg', 'both'))
    for a, b in pairs:
        r = paired(sub, a, b)
        print('    %-5s vs %-5s  acc %.3f / %.3f  只前者对 %2d 只后者对 %2d  p=%.4f  改答 %d'
              % (a, b, r['acc_a'], r['acc_b'], r['only_a_right'], r['only_b_right'],
                 r['p'], r['changed']))

    # ---- 改答率（不看对错） ----
    print('\n[4] 改答率（严格口径，不看对错）')
    for c in conds:
        ch = sum(1 for r in sub if r['out'][c]['pred'] != r['out']['base']['pred'])
        print('    %-6s 改答 %2d/%d = %.3f' % (c, ch, len(sub), ch / len(sub)))
    if len(conds) >= 2:
        for a, b in [(x, y) for i, x in enumerate(conds) for y in conds[i + 1:]]:
            pa = [r['out'][a]['pred'] != r['out']['base']['pred'] for r in sub]
            pb = [r['out'][b]['pred'] != r['out']['base']['pred'] for r in sub]
            oa = sum(1 for x, y in zip(pa, pb) if x and not y)
            ob = sum(1 for x, y in zip(pa, pb) if y and not x)
            print('    只%s改 %2d  只%s改 %2d  p=%.4f' % (a, oa, b, ob, mcnemar(oa, ob)))

    # ---- 原始口径对照（复现 H5 的错误做法，仅供对照） ----
    print('\n[5] 对照：原始口径（None 算错，即 H5 的做法）')
    for c in ['base'] + conds:
        a = sum(1 for r in recs if r['out'][c]['pred'] == r['gold']) / len(recs)
        print('    %-6s acc %.3f' % (c, a))

    if out_json:
        out_json.write_text(json.dumps({'src': src.name, 'n': len(recs), 'n_strict': len(sub),
                                        'base_acc_strict': round(base_acc, 3),
                                        'acc_strict': {c: round(sum(1 for r in sub if r['out'][c]['pred'] == r['gold']) / len(sub), 3) for c in conds},
                                        'net_strict': {c: net_of(sub, c) for c in conds},
                                        'paired_strict': {('%s_vs_%s' % (a, b)): paired(sub, a, b) for a, b in pairs}},
                                       ensure_ascii=False, indent=1), encoding='utf-8')
        print('\nwritten', out_json)


if __name__ == '__main__':
    main()
