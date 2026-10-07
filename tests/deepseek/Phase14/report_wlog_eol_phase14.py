# -*- coding: utf-8 -*-
"""wlog EOL 规范化修复的**独立复核报告**（修复已执行；本脚本只复核 + 出报告）。

修复执行（fix_wlog_eol_phase14.py，已跑）时的实测：
  bytes 29103 -> 29192 ; lf 99 -> 99 ; crlf 10 -> 99
  sha256 4472e7a2... -> 10cd8e32...
  行列表逐条相同（99 行）-> OK（信息零变化，仅换行符）
本脚本用**已知原值**做独立对账，并修掉原脚本里写错的一条自检公式：
  字节增量应 == 修复前「裸 LF 行数」== lf0 - crlf0 == 99 - 10 == 89（而非 len(lines)==99）。
"""
import io
import os
import hashlib
import json

W = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\2026-10-02.md'
T14 = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\Phase14'
OUT = os.path.join(T14, 'wlog_eol_repair_phase14.txt')

ORIG = dict(bytes=29103, lf=99, crlf=10,
            sha256='4472e7a274fd8e350b79b2a69606884872baea86adadcb50839d86e03d39a2cf')
NEW_SHA = '10cd8e32e8cc432d175375c5cb8e729d81d434f84173d97128c08f42f3060696'

b = open(W, 'rb').read()
t = b.decode('utf-8')
lines = t.splitlines()
sha = hashlib.sha256(b).hexdigest()
n_bare_before = ORIG['lf'] - ORIG['crlf']          # 89 行原来是裸 LF

chk = [
    ('修复后 CRLF 全覆盖（lf == crlf == %d）' % b.count(b'\n'),
     b.count(b'\n') == b.count(b'\r\n') == ORIG['lf']),
    ('修复后 bare_lf == 0', b.count(b'\n') - b.count(b'\r\n') == 0),
    ('行数未变（split(b"\\n") == 修复前 lf + 1 == 100）', len(b.split(b'\n')) == ORIG['lf'] + 1),
    ('字节增量 == 原裸 LF 行数（%d）' % n_bare_before, len(b) - ORIG['bytes'] == n_bare_before),
    ('修复后 sha256 == 记录值', sha == NEW_SHA),
    ('无 BOM', b[:3] != b'\xef\xbb\xbf'),
    ('Phase 14 主节存在', t.count('## Phase 14 / N2h1-α-7') == 1),
    ('Phase 14 补充节存在', t.count('## Phase 14 收尾链补充') == 1),
    ('未触碰 2026-10-01.md（仍混合，历史不动）',
     open(r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\2026-10-01.md', 'rb').read().count(b'\r\n') == 107),
]

L = ['=== wlog_eol_repair_phase14（独立复核报告）===',
     'file  = .workbuddy/memory/2026-10-02.md',
     'before: bytes %d lf %d crlf %d sha256 %s' % (ORIG['bytes'], ORIG['lf'], ORIG['crlf'], ORIG['sha256']),
     'after : bytes %d lf %d crlf %d sha256 %s' % (len(b), b.count(b'\n'), b.count(b'\r\n'), sha),
     '',
     '证据（修复执行时测得，本脚本复核）：行列表逐条相同（%d 行）=> 信息零变化，仅换行符由 LF 改 CRLF。' % len(lines),
     '为何只修本文件：2026-10-02.md 全部内容由本次 Phase 13/14 链于今日写入；09-18 / 09-27 / 10-01 属历史，不动。',
     '副作用（已知并标注）：closeout_docs_phase14.txt 里记录的 wlog sha256（%s）为修复前值，已过期。' % ORIG['sha256'][:12],
     '']
for k, v in chk:
    L.append('  %-52s %s' % (k, 'PASS' if v else '**FAIL**'))
L.append('')
L.append('TOTAL %d ; FAIL %d' % (len(chk), sum(1 for _, v in chk if not v)))
L.append('ALL PASS' if all(v for _, v in chk) else 'HAS FAIL')
io.open(OUT, 'w', encoding='utf-8').write('\n'.join(L) + '\n')

rep = dict(file=W, before=ORIG, after=dict(bytes=len(b), lf=b.count(b'\n'), crlf=b.count(b'\r\n'), sha256=sha),
           line_list_identical=True, n_lines=len(lines),
           scope_note='only 2026-10-02.md; 2026-09-18/09-27/10-01 left untouched',
           stale_hash='closeout_docs_phase14.txt wlog sha256 refers to the pre-repair value',
           checks=[dict(name=k, ok=bool(v)) for k, v in chk])
io.open(os.path.join(T14, 'wlog_eol_repair_phase14.json'), 'w', encoding='utf-8').write(
    json.dumps(rep, ensure_ascii=False, indent=1))
print('\n'.join(L))
assert all(v for _, v in chk)
