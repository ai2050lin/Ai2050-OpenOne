# -*- coding: utf-8 -*-
"""H4 分析器：对 mech_cot_L*.json 做配对检验、功效估算与分层。

为什么单独写：n=40 那轮我差点把「7.5 个点」当成结论，实际 McNemar p=0.453。
掉几个点在 n=40 上几乎没有判别力，必须固定成流程，不能每轮手算。

输出四块：
1. 各条件 acc / 格式率 / 与 base 不同题数
2. 所有两两配对的 McNemar 精确检验（因变量 = 是否正确；另附「答案是否改变」的检验）
3. 分层：base 答对 / 答错的子集里，各条件掉了多少（答错的题上「掉分」无意义，分开看才有解释力）
4. 功效估算：按观测到的 discordant 比例，要多大的 n 才能 p<0.05

用法：.venv/Scripts/python.exe -X utf8 tools/mech_cot_analyze.py tempdata/mech_cot_L34_n169.json
"""
import json
import sys
from math import comb
from pathlib import Path


def mcnemar(n10, n01):
    """精确 McNemar（双尾二项）。n10 = 只 A 对，n01 = 只 B 对。"""
    n = n10 + n01
    if n == 0:
        return 1.0
    k = min(n10, n01)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return min(1.0, p)


def power_note(n10, n01, n_cur):
    """按当前不一致比例，估算达到 p<0.05 所需的 n。"""
    n = n10 + n01
    if n == 0:
        return None
    r = min(n10, n01) / n                     # 少数派占比
    if abs(0.5 - r) < 1e-6:
        return None
    for need in range(n_cur, 20001, 5):
        disc = int(round(n / n_cur * need))
        k = int(round(disc * r))
        if k < 1 or k >= disc:
            continue
        if mcnemar(k, disc - k) < 0.05:
            return need
    return None


def main():
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('tempdata/mech_cot_L34_n169.json')
    d = json.loads(p.read_text(encoding='utf-8'))
    recs = d['records']
    conds = list(recs[0]['out'])
    n = len(recs)
    print('文件 %s   层 L%s   n=%d   条件 %s' % (p.name, d.get('block'), n, conds))

    def ok(r, c):
        return r['out'][c]['pred'] == r['gold']

    print('\n[1] 各条件')
    for c in conds:
        acc = sum(ok(r, c) for r in recs) / n
        fmt = sum(r['out'][c]['pred'] is not None for r in recs) / n
        chg = sum(r['out'][c]['pred'] != r['out']['base']['pred'] for r in recs)
        print('  %-8s acc %.3f (%d/%d)  格式 %.3f  与base不同 %d' % (c, acc, sum(ok(r, c) for r in recs), n, fmt, chg))

    print('\n[2] 配对 McNemar（因变量=是否正确）')
    for i, a in enumerate(conds):
        for b in conds[i + 1:]:
            n10 = sum(1 for r in recs if ok(r, a) and not ok(r, b))
            n01 = sum(1 for r in recs if (not ok(r, a)) and ok(r, b))
            print('  %-8s vs %-8s  只%s对 %2d  只%s对 %2d  p=%.4f %s'
                  % (a, b, a, n10, b, n01, mcnemar(n10, n01),
                     '显著' if mcnemar(n10, n01) < 0.05 else ''))

    print('\n[3] 配对（因变量=答案是否改变，与 gold 无关）')
    for a in conds:
        if a == 'base':
            continue
        chg = sum(1 for r in recs if r['out'][a]['pred'] != r['out']['base']['pred'])
        print('  %-8s 改答 %d/%d' % (a, chg, n))

    print('\n[4] 分层（base 对 / base 错）')
    right = [r for r in recs if ok(r, 'base')]
    wrong = [r for r in recs if not ok(r, 'base')]
    for lab, grp in (('base 对', right), ('base 错', wrong)):
        if not grp:
            continue
        line = '  %-8s n=%3d  ' % (lab, len(grp))
        for c in conds:
            line += '%s %.3f  ' % (c, sum(ok(r, c) for r in grp) / len(grp))
        print(line)

    print('\n[5] 功效（当前 discordant 比例 → 所需 n）')
    for i, a in enumerate(conds):
        for b in conds[i + 1:]:
            n10 = sum(1 for r in recs if ok(r, a) and not ok(r, b))
            n01 = sum(1 for r in recs if (not ok(r, a)) and ok(r, b))
            need = power_note(n10, n01, n)
            print('  %-8s vs %-8s discordant %2d  需要 n≈%s' % (a, b, n10 + n01, need))

    v = d.get('verdict') or {}
    print('\nverdict', v)
    print('reading', d.get('reading'))


if __name__ == '__main__':
    main()
