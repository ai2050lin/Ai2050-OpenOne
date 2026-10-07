# -*- coding: utf-8 -*-
"""Phase 13 收尾缺陷修复：rdc-phase-closeout/SKILL.md 被误改成 LF-only。

缺陷链：`patch_skills_phase13b.py` 用 `io.open(p, encoding='utf-8').read()`
（universal newline ⇒ CRLF 折成 LF）+ `write(..., newline='')` ⇒ **整文件 LF 化**
（实测 bare_LF 0 -> 114，字节数少 114 = 行数）。同目录另两个技能仍是 CRLF。

本脚本：
  1) 断言当前确为 pure-LF；
  2) 在「环境陷阱速查」补一条**写侧换行符陷阱**（本缺陷本身入册）；
  3) 恢复 CRLF 并以 bytes 写盘（`open(...,'wb')`）；
  4) 逐条回读复核 + 与其他技能比对换行符一致性。

纪律：写后回读（铁律 o）；字节数/哈希一律走 bytes。
"""
import io
import os
import hashlib

P = r'C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md'
OTHERS = [r'C:\Users\Admin\.workbuddy\skills\rdc-main-axis-probe\SKILL.md',
          r'C:\Users\Admin\.workbuddy\skills\rdc-dual-arm-phase-template\SKILL.md']

b0 = io.open(P, 'rb').read()
sha0 = hashlib.sha256(b0).hexdigest()
n_crlf0 = b0.count(b'\r\n')
n_lf0 = b0.count(b'\n') - n_crlf0
assert n_crlf0 == 0, '前置断言失败：文件并非 pure-LF（CRLF=%d）' % n_crlf0
print('[1] 前置断言 OK：pure-LF，bare_LF=%d，bytes=%d，sha8=%s' % (n_lf0, len(b0), sha0[:8]))

text = b0.decode('utf-8')

ANCHOR = '- Edit/Write 报成功后必须 Grep/Read 复核真实磁盘。\n'
assert text.count(ANCHOR) == 1, 'ANCHOR count=%d' % text.count(ANCHOR)

NEW = (
    '- **Python 补丁脚本读写文本文件必须显式保留换行符（2026-10-02 Phase 13 实测）**：'
    r'本技能的 `SKILL.md` 与同目录另两个技能同为 **CRLF**（实测 151 / 58 个 CRLF）；'
    r'若补丁脚本用 `io.open(p, encoding="utf-8").read()`（universal newline 把 `\r\n` 折成 `\n`）'
    r'再 `write(..., newline="")`，**整文件会被静默改写成 LF**'
    r'（本次实测 `rdc-phase-closeout` 的 `bare_LF` 由 0 变 114、字节数少掉恰好 114 = 行数，'
    r'另两个技能未受影响 ⇒ 纯属本脚本缺陷）。**对策**：改技能 / 日记类文本文件时，'
    r'① 读用 `open(p,"rb").read()` 或 `io.open(p, newline="")`；'
    r'② 写用 `open(p,"wb")` 写 bytes（或文本已含 `\r\n` 时 `write(text, newline="")`）；'
    r'③ **改完必须探一次 `CRLF` / `bare_LF` 计数并与其他同类文件比对**，'
    r'字节数差 = 行数即是本坑的指纹。\n'
)

text2 = text.replace(ANCHOR, ANCHOR + NEW)
b2 = text2.replace('\n', '\r\n').encode('utf-8')
io.open(P, 'wb').write(b2)

# ---------- 回读复核 ----------
rb = io.open(P, 'rb').read()
n_crlf = rb.count(b'\r\n')
n_lf = rb.count(b'\n') - n_crlf
assert n_lf == 0, '回读失败：仍有 bare_LF=%d' % n_lf
assert rb.replace(b'\r\n', b'\n') == text2, '回读失败：内容不一致'
assert rb.count('写侧换行符'.encode('utf-8')) == 0  # 标题里没这个说法，防误配
assert 'bare_LF` 由 0 变 114'.encode('utf-8') in rb
assert rb.count(NEW.replace('\n', '\r\n').encode('utf-8')) == 1
# 与其他技能换行符一致
for q in OTHERS:
    qb = io.open(q, 'rb').read()
    assert qb.count(b'\n') - qb.count(b'\r\n') == 0, '同类文件 %s 非 CRLF' % q

print('[2] 已补「写侧换行符陷阱」1 条（count==1 回读通过）')
print('[3] CRLF 恢复：CRLF %d -> %d，bare_LF %d -> %d' % (n_crlf0, n_crlf, n_lf0, n_lf))
print('    bytes %d -> %d（+%d = 新增行数 %d + 新增字节）' % (len(b0), len(rb), len(rb) - len(b0), len(rb) - len(b0)))
print('    sha8 %s -> %s' % (sha0[:8], hashlib.sha256(rb).hexdigest()[:8]))
print('    与另两个技能换行符一致 OK')
print('DONE')
