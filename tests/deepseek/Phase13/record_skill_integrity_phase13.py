# -*- coding: utf-8 -*-
"""Phase 13 收尾末环：把「技能补丁的换行符副作用」全程证据落盘 + 追加 wlog。

证据链（三条互校）：
  E1 真身基准（本轮首次探测，独立记录）：bytes=21126 / sha8=d6bc8441
  E2 剥除本轮两处改动 + 还原头部 14→15 ⇒ bytes=21126 / sha8=d6bc8441（逐字节命中）
  E3 三个技能换行符一致：closeout CRLF / probe CRLF / template CRLF，bare_LF 全 0
"""
import io
import os
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SK = r'C:\Users\Admin\.workbuddy\skills'
P = os.path.join(SK, 'rdc-phase-closeout', 'SKILL.md')

cur = io.open(P, 'rb').read()
A = '磁盘。\r\n- **Python 补丁脚本读写文本文件'.encode('utf-8')
B = b'\r\n- GPU'
s = cur.find(A) + len('磁盘。\r\n'.encode('utf-8'))
e = cur.find(B)
b = cur[:s] + cur[e + 2:]
j0 = b.find('15. **勘误触发'.encode('utf-8'))
j2 = b.find('## 参照实现'.encode('utf-8'))
b = b[:j0] + b[j2:]
b2 = b.replace('实测的 15 条收尾教训'.encode('utf-8'), '实测的 14 条收尾教训'.encode('utf-8'))
hit = (len(b2) == 21126 and hashlib.sha256(b2).hexdigest()[:8] == 'd6bc8441')
assert hit, '还原未命中真身'

lines = []
lines.append('Phase 13 / N2h1-α-6  收尾链末环：技能补丁副作用取证')
lines.append('=' * 72)
lines.append('')
lines.append('【缺陷】patch_skills_phase13b.py 用 io.open(p, encoding="utf-8").read()')
lines.append('        （universal newline ⇒ CRLF 折成 LF）+ write(..., newline="")')
lines.append('        ⇒ rdc-phase-closeout/SKILL.md 整文件被静默改写成 LF-only。')
lines.append('        指纹：bare_LF 0->114，字节数少掉恰好 114 = 行数；')
lines.append('        同目录另两个技能未受影响（纯属该脚本缺陷）。')
lines.append('')
lines.append('【二次缺陷】同一脚本把新条目的行尾写成 raw string r"...\\n"')
lines.append('        ⇒ 落成「字面量反斜杠+n」两字符，新条目与下一行粘成同一 markdown 行。')
lines.append('')
lines.append('【修复】fix_skill_eol_phase13.py   ：恢复 CRLF + 补「写侧换行符陷阱」条目')
lines.append('        fix_skill_eol_phase13b.py  ：字面量反斜杠+n -> 真实 CRLF + 补第 ④ 条说明')
lines.append('')
lines.append('【证据 E1】真身基准（本轮首次探测，独立记录）')
lines.append('           bytes = 21126   sha8 = d6bc8441')
lines.append('【证据 E2】从当前文件剥除本轮两处改动、并把头部 15->14 还原：')
lines.append('           bytes = %d   sha8 = %s   CRLF = %d   bare_LF = %d'
             % (len(b2), hashlib.sha256(b2).hexdigest()[:8], b2.count(b'\r\n'),
                b2.count(b'\n') - b2.count(b'\r\n')))
lines.append('           ⇒ 与 E1 逐字节相同（sha8 d6bc8441）⇒ 两处改动均为纯增量，原内容零损失。')
lines.append('')
lines.append('【证据 E3】三技能换行符一致性（改后）：')
for name in ['rdc-phase-closeout', 'rdc-main-axis-probe', 'rdc-dual-arm-phase-template']:
    q = os.path.join(SK, name, 'SKILL.md')
    qb = io.open(q, 'rb').read()
    lines.append('           %-28s bytes=%-6d CRLF=%-4d bare_LF=%-3d sha8=%s'
                 % (name, len(qb), qb.count(b'\r\n'),
                    qb.count(b'\n') - qb.count(b'\r\n'), hashlib.sha256(qb).hexdigest()[:8]))
lines.append('')
lines.append('【入册】（rdc-phase-closeout 技能 / 环境陷阱速查）')
lines.append('           新增 1 条「Python 补丁脚本读写文本文件必须显式保留换行符」＋第 ④ 点')
lines.append('           「补丁文本源码里的换行别写成 raw string 的 \\n」，本坑自身入册。')
lines.append('')
lines.append('【结论】技能当前状态：bytes=%d / CRLF=%d / bare_LF=%d / sha8=%s'
             % (len(cur), cur.count(b'\r\n'), cur.count(b'\n') - cur.count(b'\r\n'),
                hashlib.sha256(cur).hexdigest()[:8]))
lines.append('        15 条教训（编号 1..15 各 1 次）+ 环境陷阱条目 count==1，均已回读复核。')
lines.append('')

out = os.path.join(ROOT, 'tests', 'deepseek', 'Phase13', 'skill_integrity_phase13.txt')
io.open(out, 'w', encoding='utf-8', newline='\r\n').write('\n'.join(lines))
print('[1] 证据落盘 -> %s (%d B)' % (out, len(io.open(out, 'rb').read())))
print('    E2 命中真身 d6bc8441 :', hit)

# ---- wlog 追加 ----
wl = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-02.md')
before = io.open(wl, 'rb').read()
note = (
    '\r\n- **技能补丁副作用取证（本轮自捕获，已修复并逐字节证明零损失）**：'
    '`patch_skills_phase13b.py` 用 universal-newline 读 + `newline=""` 写，'
    '把 `rdc-phase-closeout/SKILL.md` **整文件静默改写成 LF-only**（bare_LF 0→114、字节数少掉恰好 114 = 行数；'
    '同目录另两技能未受影响）；同一脚本还把新条目行尾写成 raw string `' + chr(92) + 'n`，'
    '落成字面量两字符、把新条目与下一行粘成一行。'
    '修复：`fix_skill_eol_phase13.py` 恢复 CRLF + 补「写侧换行符陷阱」条目；'
    '`fix_skill_eol_phase13b.py` 换真 CRLF + 补第 ④ 点说明。'
    '**证据**：从当前文件剥除本轮两处改动、头部 15→14 还原后 '
    '`bytes=21126 / sha8=d6bc8441`，与本轮**首次探测的独立记录逐字节相同** ⇒ 原内容零损失；'
    '三技能换行符现已一致（CRLF，bare_LF 全 0）。记录 → `tests/deepseek/Phase13/skill_integrity_phase13.txt`。\r\n'
)
io.open(wl, 'ab').write(note.encode('utf-8'))
after = io.open(wl, 'rb').read()
assert after.startswith(before) and len(after) > len(before)
print('[2] wlog %d -> %d B ; sha256=%s' % (len(before), len(after), hashlib.sha256(after).hexdigest()))
print('DONE')
