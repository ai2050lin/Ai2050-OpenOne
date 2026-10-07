# -*- coding: utf-8 -*-
"""修正 wlog 补记段里的技能文件体积（之前用 len(s.encode()) 在文本模式 round-trip 后取值，
Windows 上 \n<->\r\n 转换使该值比真实文件小「行数」字节），并补记该工具坑。
真实值：probe 65,963 B / sha8 1506cb7e；closeout 36,850 B / sha8 34aa2a4c（均 CRLF-only）。
"""
import io, os, hashlib

P = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\2026-10-02.md'
SKP = r'C:\Users\Admin\.workbuddy\skills\rdc-main-axis-probe\SKILL.md'
SKC = r'C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md'
BT = chr(96)


def c(x):
    return BT + x + BT


def info(p):
    b = open(p, 'rb').read()
    return len(b), hashlib.sha256(b).hexdigest()[:8]


pb, pb8 = info(SKP)
cb, cb8 = info(SKC)

b0 = open(P, 'rb').read()
s = b0.decode('utf-8')

old1 = '→ 63,746 → **65,788 B**（14 臂 + **55 坑**）'
assert s.count(old1) == 1, 'a1=%d' % s.count(old1)
s = s.replace(old1, '→ 63,746 → **%s B**（14 臂 + **55 坑**；sha8 %s）' % (format(pb, ','), pb8))

old2 = '→ 31,963 → **34,893 B**（**24 教训**）'
assert s.count(old2) == 1, 'a2=%d' % s.count(old2)
s = s.replace(old2, '→ 31,963 → **%s B**（**25 教训**；sha8 %s）' % (format(cb, ','), cb8))

AN = '- **下一步**：P16 判据冻结后开跑；'
assert s.count(AN) == 1, 'a3=%d' % s.count(AN)
INS = ('- **工具坑（体积上报）**：修补脚本里用 ' + c('len(s.encode("utf-8"))') + ' 报「bytes」在 Windows 文本模式下**偏小** —— '
       '读时 ' + c('\\r\\n -> \\n') + ' 折叠、写时 ' + c('\\n -> \\r\\n') + ' 展开，字符串长度比磁盘真实字节少「行数」字节'
       '（本轮 probe 偏小 175 B、closeout 偏小 144 B）。**对策**：体积/校验一律用 ' + c('os.path.getsize()') + ' 或**二进制读**取 '
       + c('hashlib.sha256(open(p,"rb").read())') + '；本段已按真实值更正。\n')
s = s.replace(AN, INS + AN)

b1 = s.replace('\r\n', '\n').replace('\n', '\r\n').encode('utf-8')
open(P, 'wb').write(b1)
b2 = open(P, 'rb').read()
assert b2 == b1
d = b2.decode('utf-8')
print('wlog %d -> %d B ; bare_lf=%d ; sha256=%s'
      % (len(b0), len(b2), b2.count(b'\n') - b2.count(b'\r\n'), hashlib.sha256(b2).hexdigest()))
print('probe real = %d / %s ; closeout real = %d / %s' % (pb, pb8, cb, cb8))
print('ok =', (format(pb, ',') + ' B') in d and (format(cb, ',') + ' B') in d and '工具坑（体积上报）' in d)
