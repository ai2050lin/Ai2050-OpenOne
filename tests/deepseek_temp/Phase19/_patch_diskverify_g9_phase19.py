# -*- coding: utf-8 -*-
"""修正 disk_verify_phase19.py 的 G9（作用域错误：Ledger 混有 N 线/G 线/无 phase 条目）
与 H10 的显示（打印 0-based 索引）。"""
import io
import os

P = os.path.join(r'D:\AI2050\Ai2050-OpenOne', 'tests', 'deepseek', 'Phase19', 'disk_verify_phase19.py')
s = io.open(P, encoding='utf-8').read()

old9 = ("chk('G9 ledger 无重复 phase', len({m.get('phase') for m in LG['measurements']}) == len(LG['measurements']),\n"
        "    'phases=%d uniq=%d' % (len(LG['measurements']), len({m.get('phase') for m in LG['measurements']})))")
new9 = ("_nl = [m for m in LG['measurements'] if str(m.get('name', '')).startswith('n2h1a')]\n"
        "chk('G9 N 线条目相位唯一且 = 8..19', sorted(m['phase'] for m in _nl) == list(range(8, 20)),\n"
        "    'N-line=%d phases=%s' % (len(_nl), sorted(m['phase'] for m in _nl)))")
assert s.count(old9) == 1, 'old9 count=%d' % s.count(old9)
s = s.replace(old9, new9)

old10 = "'line=%s of %d' % (i19, len(mt))"
new10 = "'line=%d of %d' % (i19[0] + 1, len(mt))"
assert s.count(old10) == 1, 'old10 count=%d' % s.count(old10)
s = s.replace(old10, new10)

io.open(P, 'w', encoding='utf-8', newline='\n').write(s)
s2 = io.open(P, encoding='utf-8').read()
assert new9.splitlines()[0] in s2 and new10 in s2
print('PATCH OK  bytes=%d' % len(s2.encode('utf-8')))
