# -*- coding: utf-8 -*-
"""把 wlog 补记段里的 `TOTAL checks = 122` 更新为「122（首跑）→ 126（本轮扩 I6b–I6e）」，
并补记 bash 反引号吞字事故与回滚验证。CRLF 写盘 + 复核。"""
import io, hashlib

P = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\2026-10-02.md'
b0 = open(P, 'rb').read()
s = b0.decode('utf-8')
BT = chr(96)


def c(x):
    return BT + x + BT


old = c('TOTAL checks = 122 ; FAIL = 0') + '**（含新增分区 J：'
new = (c('TOTAL checks = 122 ; FAIL = 0') + '（首跑）→ **' + c('TOTAL checks = 126 ; FAIL = 0') +
       '**（本轮为「wlog 自身完整性」在分区 I 扩了 `I6b/I6c/I6d/I6e` 四项后）**（含新增分区 J：')
assert s.count(old) == 1, 'anchor1 count=%d' % s.count(old)
s = s.replace(old, new)

AN = '- **工具教训（本轮再次踩中）**'
assert s.count(AN) == 1, 'anchor2 count=%d' % s.count(AN)
INS = ('- **wlog 追加事故与回滚验证（同一轮内）**：收尾补记段**首版**经 bash 内联 ' + c('python -c') +
       ' 写入，段内 ' + c('`code`') + ' 片段被 bash 做**命令替换**并静默替换为空 ⇒ 段内所有代码标识符消失'
       '（文件从 38,233 → 41,305 B，看似「写成功」）。**检测**：统计段内反引号个数（污染版 = 0，正常版 = 80）。'
       '**回滚**：切到段头之前，并断言切出的前缀 ' + c('sha256 = 8f69229b...c842c3 ; 38,233 B') +
       ' —— 与 ' + c('closeout_docs_phase15.py') + ' 报出的上一版 sha256 逐字节一致，证明回滚点就是已复核状态；'
       '随后用**编辑器写盘的 .py 文件**（不经 bash）重写该段，并新增磁盘复核项 ' + c('I6d') +
       '（段内含反引号）作为**长期不变量**。\n')
s = s.replace(AN, INS + AN)

b1 = s.replace('\r\n', '\n').replace('\n', '\r\n').encode('utf-8')
open(P, 'wb').write(b1)
b2 = open(P, 'rb').read()
assert b2 == b1, 'disk readback mismatch'
d = b2.decode('utf-8')
print('wlog %d -> %d B ; bare_lf=%d' % (len(b0), len(b2), b2.count(b'\n') - b2.count(b'\r\n')))
print('sha256 = %s' % hashlib.sha256(b2).hexdigest())
print('has 126 =', 'TOTAL checks = 126' in d, '| has incident =', 'wlog 追加事故与回滚验证' in d)
