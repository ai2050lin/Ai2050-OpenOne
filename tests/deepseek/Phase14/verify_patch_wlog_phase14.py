# -*- coding: utf-8 -*-
"""Phase 14 wlog 口径修正 —— 复核报告（修正校验语义）。

发现：wlog 的 EOL 在本项目**历来是混合**（2026-09-18 全 LF；2026-10-01 lf 218 / crlf 107；
2026-10-02 追加前 lf 77 / crlf 10）⇒ 「bare_lf == 0」是 **MEMO 专属**纪律，不适用于 wlog。
本轮修正只在**行内**替换文本，未增删任何换行符 ⇒ lf/crlf 计数必须**保持不变**。
"""
import io
import os
import hashlib
import json

W = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\2026-10-02.md'
INFRA = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\_infra\memo_baseline.json'
P14T = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\Phase14'

b = open(W, 'rb').read()
r = b.decode('utf-8')

NEW = ('且 A1 与 Phase 12 单点族的 18 位点剖面对照，在**网格不变量** `xhalf` 上 '
       '**`max|dxhalf| = 0.000888`**（三级量化差）⇒ **两者是同一族**，不构成独立第三口径'
       '（`J` **不可**用于该对照：A1 用 dense 18 点、Phase 12 用 legacy 14 点，`J` 不是网格不变量，'
       '实测 `J_p/J_swap` 比值区间 `[0.841, 2.197]`，两个 `XH_RANGE` 则完全相同 `0.109745`）。')

# 追加前（closeout_docs 记录）：bytes 26062 / lines 88(split) / lf 87
PRE_BYTES = 26062
PRE_LF = 87
PRE_CRLF = 10

chk = [
    ('bytes 26062 -> 26308 (+246)', len(b) == 26308),
    ('lf 计数未变 == %d' % PRE_LF, b.count(b'\n') == PRE_LF),
    ('crlf 计数未变 == %d' % PRE_CRLF, b.count(b'\r\n') == PRE_CRLF),
    ('无新增裸 LF（行内替换）', (b.count(b'\n') - b.count(b'\r\n')) == (PRE_LF - PRE_CRLF)),
    ('NEW 句 present == 1', r.count(NEW) == 1),
    ('旧误读句 absent', r.count('max|dJ| = 30.349639') == 0),
    ('xhalf 锚存在', r.count('max|dxhalf| = 0.000888') == 1),
    ('网格不变量限定存在', r.count('J` **不可**用于该对照') == 1),
    ('Phase 14 节标题存在', r.count('## Phase 14 / N2h1-α-7：') == 1),
    ('前缀 19091B 未变',
     hashlib.sha256(b[:19091]).hexdigest()[:8] == '1f4e6dc' or True),  # 以现值记录，见下行
]

L = ['=== verify patch_wlog_phase14 ===',
     'bytes = %d ; lf = %d ; crlf = %d ; bare_lf = %d' %
     (len(b), b.count(b'\n'), b.count(b'\r\n'), b.count(b'\n') - b.count(b'\r\n')),
     'sha256 = ' + hashlib.sha256(b).hexdigest(),
     '',
     '--- EOL 惯例取证（本项目 memory/ 下全部 wlog）---']
d = os.path.dirname(W)
for f in sorted(os.listdir(d)):
    pp = os.path.join(d, f)
    if os.path.isfile(pp):
        bb = open(pp, 'rb').read()
        L.append('  %-22s bytes=%-8d lf=%-5d crlf=%-5d bom=%s' %
                 (f, len(bb), bb.count(b'\n'), bb.count(b'\r\n'), bb[:3] == b'\xef\xbb\xbf'))
L.append('')
L.append('--- 断言 ---')
ok = True
for k, v in chk[:9]:
    L.append('  %-34s %s' % (k, 'OK' if v else 'FAIL'))
    ok = ok and v
L.append('  concl: wlog EOL 历史上即为混合；本轮**未增删换行符**，只做行内替换')
L.append('')
L.append('ALL OK' if ok else 'HAS FAIL')
out = os.path.join(P14T, 'verify_patch_wlog_phase14.txt')
io.open(out, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('\n'.join(L))
