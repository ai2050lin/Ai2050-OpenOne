# -*- coding: utf-8 -*-
"""补丁 6：closeout_docs_phase20.py
 (d1) `q(PID,'xhalf')` 取到 dict ⇒ 改取 max_abs_dxh。
 (d2) 确认集 Δcom_B 直接索引 `com_B_conf['INC_ALL']`，该值可能为 None ⇒ 加 dl() 守卫。
 (d3) `not PC[k]['pass_']` 会把 P10（pass_=None，描述性）误列为「否证」⇒ 改 `is False`。
 (d4) 「Ledger 补登 N 线第 12 条」硬编码错（P8–P20 共 13 条）⇒ 由 Ledger 现场计数渲染。
 (d5) 「19 处 segfault」措辞失真 ⇒ 改「加载至约 19pct 权重处 segfault」。
"""
import io

P = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase20\closeout_docs_phase20.py'
s = io.open(P, encoding='utf-8').read()
n = 0


def rep(old, new):
    global s, n
    c = s.count(old)
    assert c == 1, 'count=%d :: %r' % (c, old[:70])
    s = s.replace(old, new); n += 1


# (d2) 先加辅助函数
rep("def E(a, k):\n    return R['arms'][a]['E10_summary'].get(k)\n",
    "def E(a, k):\n    return R['arms'][a]['E10_summary'].get(k)\n"
    "\n"
    "\n"
    "def dl(a, c='INC_ALL'):\n"
    "    v = E(a, 'com_B_conf').get(c); z = E(a, 'com_B').get(c)\n"
    "    return abs(v - z) if (v is not None and z is not None) else None\n")

# (d1)
rep("     q(PID, 'xhalf'), q(PID2, 'xhalf'), FL['QUANT_TOL_XHALF'],",
    "     fn(QPM[PID]['xhalf']['max_abs_dxh'], 4), fn(QPM[PID2]['xhalf']['max_abs_dxh'], 4),\n"
    "     FL['QUANT_TOL_XHALF'],")

# (d2)
rep("     fn(abs(E(A0, 'com_B_conf')['INC_ALL'] - E(A0, 'com_B')['INC_ALL']), 3),\n"
    "     fn(abs(E(A0b, 'com_B_conf')['INC_ALL'] - E(A0b, 'com_B')['INC_ALL']), 3),\n"
    "     fn(abs(E(A1, 'com_B_conf')['INC_ALL'] - E(A1, 'com_B')['INC_ALL']), 3),\n"
    "     fn(abs(E(A1b, 'com_B_conf')['INC_ALL'] - E(A1b, 'com_B')['INC_ALL']), 3), 3.0,\n",
    "     fn(dl(A0), 3), fn(dl(A0b), 3), fn(dl(A1), 3), fn(dl(A1b), 3), 3.0,\n")

# (d3)
rep("    _f = [k for k in sorted(PC) if not PC[k]['pass_']]",
    "    _f = [k for k in sorted(PC) if PC[k]['pass_'] is False]")

# (d4)
rep("  '（前缀逐字节未变、BOM/CRLF、`bare_lf 0`、Phase 标题 **%d** 个）；Ledger 补登 N 线第 12 条'\n"
    "  '（%d → **%d**，verdict `%s`，`ledger_sha256_8 = %s`）。'\n"
    "  % (p20_line[0] if p20_line else '?', PRE['bytes'], len(mb), PRE['lines'], len(mt), len(ph_lines),\n"
    "     len(LG['measurements']) - 1, len(LG['measurements']), tail['verdict'], LG['ledger_sha256_8']))",
    "  '（前缀逐字节未变、BOM/CRLF、`bare_lf 0`、Phase 标题 **%d** 个）；Ledger 补登 N 线第 %d 条'\n"
    "  '（%d → **%d**，verdict `%s`，`ledger_sha256_8 = %s`）。'\n"
    "  % (p20_line[0] if p20_line else '?', PRE['bytes'], len(mb), PRE['lines'], len(mt), len(ph_lines),\n"
    "     sum(1 for m in LG['measurements']\n"
    "         if isinstance(m.get('phase'), int) and 8 <= m['phase'] <= 20),\n"
    "     len(LG['measurements']) - 1, len(LG['measurements']), tail['verdict'], LG['ledger_sha256_8']))")

# (d5)
rep("**不参与** bf16 腿（P19 实测 bf16 加载 19 处 segfault）⇒ '",
    "**不参与** bf16 腿（P19 实测 bf16 加载至约 19pct 权重处 segfault）⇒ '")

io.open(P, 'w', encoding='utf-8', newline='\n').write(s)
print('PATCHED %d spots' % n)
for pr in ["def dl(a, c='INC_ALL')", "QPM[PID]['xhalf']['max_abs_dxh']", 'fn(dl(A0), 3)',
           "PC[k]['pass_'] is False", 'N 线第 %d 条', '19pct 权重处', 'E(A0, \'com_B_conf\')']:
    print('  chk %-38s -> %d' % (pr[:38], s.count(pr)))
