# -*- coding: utf-8 -*-
"""决定性校验：从当前技能文件里剥掉新插入的条目，应逐字节还原为补丁前的 pure-LF 文件
（bytes=22337, sha8=7a496895）。"""
import io, hashlib

P = r'C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md'
NEW_LF = (
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
cur = io.open(P, 'rb').read()
lf = cur.replace(b'\r\n', b'\n')                      # 当前内容的 LF 规范形
new_b = NEW_LF.encode('utf-8')
assert lf.count(new_b) == 1, '新条目在 LF 形中出现 %d 次' % lf.count(new_b)
pre = lf.replace(new_b, b'')
print('剥除后 bytes=%d  (期望 22337)  %s' % (len(pre), len(pre) == 22337))
print('剥除后 sha8 =%s  (期望 7a496895)  %s' % (hashlib.sha256(pre).hexdigest()[:8],
                                              hashlib.sha256(pre).hexdigest()[:8] == '7a496895'))
print('当前 bytes=%d  count_LF=%d  count_CRLF=%d' % (len(cur), cur.count(b'\n'), cur.count(b'\r\n')))
print('LF 规范形 count_LF=%d ; 新条目含 \\n 数=%d' % (lf.count(b'\n'), new_b.count(b'\n')))
print('结论:', '逐字节还原成功 ⇒ 原内容零损失' if (len(pre) == 22337 and hashlib.sha256(pre).hexdigest()[:8] == '7a496895') else '**不匹配，需人工检查**')
