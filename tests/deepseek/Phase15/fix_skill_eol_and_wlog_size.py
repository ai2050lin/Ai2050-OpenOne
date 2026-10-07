# -*- coding: utf-8 -*-
"""① 规范化 closeout 技能文件的裸 LF（插入子项时带入了 2 个 \n）；② 二进制读取得真实体积/sha8；
③ 只更新 wlog 里**closeout 技能**的体积（probe 未再变动，保持 65,963 B / 1506cb7e）。
"""
import io, hashlib

SKP = r'C:\Users\Admin\.workbuddy\skills\rdc-main-axis-probe\SKILL.md'
SKC = r'C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md'
P = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\2026-10-02.md'


def stat(p):
    b = open(p, 'rb').read()
    return len(b), hashlib.sha256(b).hexdigest()[:8], b.count(b'\r\n'), b.count(b'\n') - b.count(b'\r\n')


# ① 规范化
b = open(SKC, 'rb').read()
n_bare = b.count(b'\n') - b.count(b'\r\n')
if n_bare:
    open(SKC, 'wb').write(b.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n'))
    print('closeout: normalized %d bare LF -> CRLF' % n_bare)

pb, pb8, _, pbare = stat(SKP)
cb, cb8, _, cbare = stat(SKC)
assert pbare == 0 and cbare == 0
print('probe    : %d B sha8=%s' % (pb, pb8))
print('closeout : %d B sha8=%s' % (cb, cb8))

# ② 更新 wlog（只改 closeout 的体积与 sha8）
b0 = open(P, 'rb').read()
s = b0.decode('utf-8')
old = '→ **36,850 B**（**25 教训**；sha8 34aa2a4c）'
assert s.count(old) == 1, 'anchor count=%d' % s.count(old)
s = s.replace(old, '→ **%s B**（**25 教训**；sha8 %s）' % (format(cb, ','), cb8))

b1 = s.replace('\r\n', '\n').replace('\n', '\r\n').encode('utf-8')
open(P, 'wb').write(b1)
b2 = open(P, 'rb').read()
d = b2.decode('utf-8')
print('wlog %d -> %d B ; bare_lf=%d ; sha256=%s'
      % (len(b0), len(b2), b2.count(b'\n') - b2.count(b'\r\n'), hashlib.sha256(b2).hexdigest()))
print('probe size in wlog    :', format(pb, ',') + ' B' in d)
print('closeout size in wlog :', format(cb, ',') + ' B' in d)
