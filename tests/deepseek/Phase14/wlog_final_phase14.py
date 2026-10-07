# -*- coding: utf-8 -*-
"""Phase 14 收尾链终局：wlog 再补一行定稿状态（CRLF 写入 + 前缀锚核对）。"""
import io
import os
import hashlib

W = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\2026-10-02.md'
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\Phase14\wlog_final_phase14.txt'

b0 = open(W, 'rb').read()
SEC = (
    "- **收尾链终局状态（本轮全部完成）**：独立磁盘复核 **115 checks / FAIL = 0**（修正 4 处断言缺陷后重跑，两坐标 jumps 与置换 null 全部 `0.0e+00` 逐位）；"
    "`MEMORY.md` 重写定稿 **9,882 B / 5,832 字符 / 48 行**（原 11,621 B ⇒ 注入不再截断），25 条铁律字母 **(a)–(y)** 与 MEMO 逐条对齐（本轮曾因重写把 (h)–(n)/(o)–(s) 整体错位，已用 `fix_memory_letters_phase14.py` 回正：MUST 全 1 / BAD 全 0）；"
    "wlog EOL 规范化（`29103 → 29192 B`，行列表逐条相同）；技能 `rdc-main-axis-probe` **50 坑**、`rdc-phase-closeout` **20 教训**（新增 19 复核脚本先空跑 / 20 wlog EOL 由生成器决定）。"
)
t1 = b0.decode('utf-8').rstrip('\r\n') + '\r\n\r\n' + SEC + '\r\n'
open(W, 'wb').write(t1.encode('utf-8'))
b1 = open(W, 'rb').read()
r = b1.decode('utf-8')
chk = [
    ('前缀 %d B 逐字节未变' % len(b0), hashlib.sha256(b1[:len(b0)]).hexdigest() == hashlib.sha256(b0).hexdigest()),
    ('仍 CRLF-only（bare_lf == 0）', b1.count(b'\n') == b1.count(b'\r\n')),
    ('终局行存在', r.count('收尾链终局状态') == 1),
    ('Phase 14 三节齐全', r.count('## Phase 14 / N2h1-α-7') == 1 and r.count('## Phase 14 收尾链补充') == 1),
]
L = ['=== wlog_final_phase14 ===', 'bytes %d -> %d (%+d)' % (len(b0), len(b1), len(b1) - len(b0)),
     'sha256 = ' + hashlib.sha256(b1).hexdigest(), '']
for k, v in chk:
    L.append('  %-40s %s' % (k, 'OK' if v else '**FAIL**'))
L.append('ALL OK' if all(v for _, v in chk) else 'HAS FAIL')
io.open(OUT, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('\n'.join(L))
assert all(v for _, v in chk)
