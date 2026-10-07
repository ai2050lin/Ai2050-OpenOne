# -*- coding: utf-8 -*-
"""Phase 13 技能补丁二次修复：新条目末尾的 raw-string `\\n` 落成了「字面量反斜杠+n」。

机制：`fix_skill_eol_phase13.py` 的 NEW 末段写成 raw string `r'...指纹。\\n'`，
raw string 里的 `\\n` 是**两个字符**（反斜杠 + n），不是换行 ⇒ 新条目与下一行
（`- GPU 测试...`）被粘成同一 markdown 行。

本脚本：① 把该字面量反斜杠+n 换成真实 CRLF；② 在该条末尾补一句本坑说明；
③ 以 bytes 写盘 + 回读复核 + 与另两个技能换行符比对。
"""
import io
import os
import hashlib

P = r'C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md'
BS = chr(92)

cur = io.open(P, 'rb').read()
sha0 = hashlib.sha256(cur).hexdigest()

OLD = '指纹。'.encode('utf-8') + BS.encode('ascii') + b'n' + '- GPU'.encode('utf-8')
assert cur.count(OLD) == 1, 'OLD count=%d' % cur.count(OLD)

EXTRA = (
    '④ **补丁文本源码里的换行别写成 raw string 的 `' + BS + 'n`**——'
    '那是「反斜杠 + n」两字符，会把新条目与下一行**粘成同一行**（本轮首次插入即踩此坑，'
    '已在本条自身修复）；补丁脚本里表示换行请用非 raw 字符串的 `' + BS + 'n` 或真实换行。'
)
NEW = '指纹。'.encode('utf-8') + EXTRA.encode('utf-8') + b'\r\n' + '- GPU'.encode('utf-8')

cur2 = cur.replace(OLD, NEW)
# 本文件原有多处字面量反斜杠+n（如 `\r\n` 折成 `\n` 的说明）；本次应为「移除 1（粘连处）+ 新增 2（EXTRA 里两处示例）」
BN = BS.encode('ascii') + b'n'
assert cur2.count(BN) == cur.count(BN) + 1, '反斜杠+n 计数 %d -> %d' % (cur.count(BN), cur2.count(BN))
assert (BN + b'- GPU') not in cur2, '粘连未消除'
io.open(P, 'wb').write(cur2)

# ---------- 回读 ----------
rb = io.open(P, 'rb').read()
n_crlf = rb.count(b'\r\n')
n_lf = rb.count(b'\n') - n_crlf
assert n_lf == 0, '回读：bare_LF=%d' % n_lf
assert rb.count(EXTRA.encode('utf-8')) == 1
assert (BN + b'- GPU') not in rb, '粘连未消除'
assert rb.endswith(b'\r\n')
# 新条目与下一行已分行
assert ('指纹。'.encode('utf-8') + EXTRA.encode('utf-8') + b'\r\n- GPU') in rb
# 新条目是独立的一行开头
assert rb.count('\r\n- **Python 补丁脚本读写文本文件'.encode('utf-8')) == 1
for q in [r'C:\Users\Admin\.workbuddy\skills\rdc-main-axis-probe\SKILL.md',
          r'C:\Users\Admin\.workbuddy\skills\rdc-dual-arm-phase-template\SKILL.md']:
    qb = io.open(q, 'rb').read()
    assert qb.count(b'\n') - qb.count(b'\r\n') == 0, '同类文件 %s 非 CRLF' % q

print('[1] 字面量反斜杠+n 已换成 CRLF（count==1）')
print('[2] 已补第 ④ 条说明（本坑自身入册）')
print('[3] CRLF=%d / bare_LF=%d ; bytes %d -> %d ; sha8 %s -> %s'
      % (n_crlf, n_lf, len(cur), len(rb), sha0[:8], hashlib.sha256(rb).hexdigest()[:8]))
print('    行数 =', n_crlf)
print('    与另两个技能换行符一致 OK')
print('DONE')
