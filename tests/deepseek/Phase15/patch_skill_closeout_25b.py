# -*- coding: utf-8 -*-
"""给 rdc-phase-closeout 的教训 25 追加一条子项：修补脚本上报「bytes」的坑（文本模式 round-trip）。"""
import io, hashlib

P = r'C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md'
s = io.open(P, encoding='utf-8', newline='').read()
b0 = hashlib.sha256(s.encode('utf-8')).hexdigest()
BT = chr(96)


def c(x):
    return BT + x + BT


ANCHOR = '## 参照实现（Phase 3125，210/210 全绿；Phase 3128，19/19 全绿）'
assert s.count(ANCHOR) == 1, 'anchor=%d' % s.count(ANCHOR)

SUB = ('    - **子坑：别用 ' + c('len(s.encode())') + ' 报「文件体积」** —— Windows 文本模式读时 ' + c('\\r\\n -> \\n') +
       ' 折叠、写时展开，于是字符串长度比磁盘真实字节**少「行数」字节**（Phase 15：probe 偏小 175 B、closeout 偏小 144 B，'
       '但内容其实完全正确）。**对策**：体积与校验一律 ' + c('os.path.getsize(p)') + ' 或 '
       + c('hashlib.sha256(open(p, "rb").read())') + '；凡是 wlog/MEMO 里要引用某个文件的字节数，'
       '都用**二进制读**得到的值，否则会出现「同一文件在两个地方两个大小」的不一致。\n\n')

s = s.replace(ANCHOR, SUB + ANCHOR)
io.open(P, 'w', encoding='utf-8', newline='').write(s)
b = open(P, 'rb').read()
assert b.decode('utf-8') == s, 'disk readback mismatch'
print('skill bytes = %d ; crlf=%d ; bare_lf=%d ; sha8=%s'
      % (len(b), b.count(b'\r\n'), b.count(b'\n') - b.count(b'\r\n'), hashlib.sha256(b).hexdigest()[:8]))
print('sub present:', '子坑：别用' in s)
