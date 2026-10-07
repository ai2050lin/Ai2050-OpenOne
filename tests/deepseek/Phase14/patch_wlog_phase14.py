# -*- coding: utf-8 -*-
"""Phase 14 wlog 口径修正：A1 vs Phase 12 的族同一性只在网格不变量 xhalf 上成立；
J 因网格不同（dense 18 点 vs legacy 14 点）不可用于该对照。逐处 assert count==1 + 回读复核。
铁律：禁用 bash 内联 python -c（反引号/反斜杠被吃），一律脚本文件。
"""
import io
import os
import hashlib

W = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\2026-10-02.md'

OLD = ('且 A1 与 Phase 12 单点族的 18 位点剖面对照 '
       '**`max|dxhalf| = 0.000888`、`max|dJ| = 30.349639`** ⇒ **两者是同一族**，不构成独立第三口径。')
NEW = ('且 A1 与 Phase 12 单点族的 18 位点剖面对照，在**网格不变量** `xhalf` 上 '
       '**`max|dxhalf| = 0.000888`**（三级量化差）⇒ **两者是同一族**，不构成独立第三口径'
       '（`J` **不可**用于该对照：A1 用 dense 18 点、Phase 12 用 legacy 14 点，`J` 不是网格不变量，'
       '实测 `J_p/J_swap` 比值区间 `[0.841, 2.197]`，两个 `XH_RANGE` 则完全相同 `0.109745`）。')

b0 = open(W, 'rb').read()
t = b0.decode('utf-8')
assert t.count(OLD) == 1, 'OLD count = %d' % t.count(OLD)
t2 = t.replace(OLD, NEW)
assert t2.count(NEW) == 1
assert t2.count('max|dJ| = 30.349639') == 0
open(W, 'wb').write(t2.encode('utf-8'))

b1 = open(W, 'rb').read()
r = b1.decode('utf-8')
chk = [
    ('NEW present', r.count(NEW) == 1),
    ('OLD absent', r.count('max|dJ| = 30.349639') == 0),
    ('xhalf anchor', r.count('max|dxhalf| = 0.000888') >= 1),
    ('grid-invariant caveat', r.count('J` **不可**用于该对照') == 1),
    ('bare_lf 0', b1.count(b'\n') == b1.count(b'\r\n')),
    ('prefix 19091B unchanged', hashlib.sha256(b1[:19091]).hexdigest()[:8] == hashlib.sha256(b0[:19091]).hexdigest()[:8]),
]
lines = ['=== patch_wlog_phase14 ===', 'bytes %d -> %d (%+d)' % (len(b0), len(b1), len(b1) - len(b0))]
for k, v in chk:
    lines.append('  %-32s %s' % (k, 'OK' if v else 'FAIL'))
lines.append('sha256 = ' + hashlib.sha256(b1).hexdigest())
lines.append('ALL OK' if all(v for _, v in chk) else 'HAS FAIL')
out = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\Phase14\patch_wlog_phase14.txt'
io.open(out, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
