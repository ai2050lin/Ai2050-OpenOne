# -*- coding: utf-8 -*-
"""终极完整性校验 v2：把本轮对 closeout 技能的两处改动（教训 15 + 环境陷阱新条）全部剥除，
应逐字节还原补丁前真身 —— bytes=21126 / sha8=d6bc8441（本轮首次探测独立记录）。
若通过 ⇒ 两处都是纯增量、零损失、零改写。"""
import io, hashlib

P = r'C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md'
cur = io.open(P, 'rb').read()


def strip(b):
    # (1) 剥掉「环境陷阱速查」新条（含其行尾 CRLF）
    A = '磁盘。\r\n- **Python 补丁脚本读写文本文件'.encode('utf-8')
    B = b'\r\n- GPU'
    assert b.count(A) == 1 and b.count(B) == 1
    s = b.find(A) + len('磁盘。\r\n'.encode('utf-8'))
    e = b.find(B)
    bullet = b[s:e]
    b = b[:s] + b[e + 2:]          # 连同 bullet 的行尾 CRLF 一起剥除（否则多留一个空行 = +2 B）
    # (2) 剥掉「教训 15」整段
    j0 = b.find('15. **勘误触发'.encode('utf-8'))
    j2 = b.find('## 参照实现'.encode('utf-8'))
    assert j0 != -1 and j2 != -1 and j0 < j2
    lesson = b[j0:j2]
    b = b[:j0] + b[j2:]
    return b, bullet, lesson


cand, bullet, lesson = strip(cur)
h = hashlib.sha256(cand).hexdigest()
ok = (len(cand) == 21126 and h[:8] == 'd6bc8441')
print('当前  bytes=%-6d CRLF=%-4d bare_LF=%-3d sha8=%s'
      % (len(cur), cur.count(b'\r\n'), cur.count(b'\n') - cur.count(b'\r\n'), hashlib.sha256(cur).hexdigest()[:8]))
print('剥除后 bytes=%-6d CRLF=%-4d bare_LF=%-3d sha8=%s' % (
    len(cand), cand.count(b'\r\n'), cand.count(b'\n') - cand.count(b'\r\n'), h[:8]))
print('真身   bytes=21126  CRLF=113  sha8=d6bc8441')
print('剥除量 陷阱条=%d B / 教训15=%d B / 合计 %d (+CRLF 2) = %d'
      % (len(bullet), len(lesson), len(bullet) + len(lesson), len(cur) - len(cand)))
print('⇒ 逐字节还原:', ok)
assert ok, '不匹配，需人工检查'
print('OK —— 两处改动均为纯增量，原内容零损失')
