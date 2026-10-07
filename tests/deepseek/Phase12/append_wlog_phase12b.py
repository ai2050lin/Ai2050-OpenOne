# -*- coding: utf-8 -*-
"""Phase 12 wlog 补记（磁盘复核 + 口径澄清 + 时钟事件 + 技能/MEMORY 同步）。append-only。"""
import os, io, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
WLOG = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-02.md')
MEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')

b0 = open(WLOG, 'rb').read()
mem = open(MEM, 'rb').read()
sec = """
## Phase 12 收尾补记：独立磁盘复核 + 口径澄清 + 时钟事件（00:29）

- **独立磁盘复核**：`tests/deepseek/Phase12/disk_verify_phase12.py` → `disk_verify_phase12.txt`，**49 个分区 / 0 FAIL**。覆盖：A0 文件与 sha8、A1 冻结件 sha（result/exec/seal/amend1）、A2 常量、A3 时钟、B 逐对自洽（**18 位点 × 14 α 全网格** `max|d| = 5.329e-15`）、C 曲线重算（`xhalf` / `recover` / `J_swap` 的 `max|d| = 0.000e+00`、`x_star` logistic 同格命中）、D G 族 17 项布尔、**E bootstrap 同 seed 复现全部带（18 位点 × {xhalf, J}）**、**F 置换零假设 2000 个值逐位复现 `max|d| = 0.000e+00`**、G floors（F1/F6/F10/F11/F12/F13）、H 离流形 α、I 文档落点（MEMO 273,579 B / 2858 行 / 12 标题、baseline 对账、Ledger 295）、J 判决一致性。
- **口径澄清（复核抓出，重要）**：冻结的 `recover.span` 定义是 **端点差** `recover(L34) − recover(L6) = +0.004763`，**不是**极差 `max − min = +0.005499`（两者差 15%%）。凡引用「span 0.0048」必须同时说明这是端点差 —— 否则会被误当成"整个剖面的动态范围"。
- **时钟事件（登记在案）**：`execution_phase12.json` 的 `frozen_at = 2026-10-02 00:50`，而其 mtime = `00:14:50` ⇒ **晚了 35 min，属元数据笔误**。exec 已被 `result.exec_sha8 = 67f38c53` 锚定，**不改动冻结件**（与 Phase 8 节标题 `[22:05]` 晚于 mtime 的历史异常同类）。凡以时间戳推断先后，一律与产物 mtime 交叉核对。
- **技能同步（逐处 `assert count==1` + 16 项回读全绿）**：`rdc-main-axis-probe` 12 → **13 臂**（新增 N2h1-α-5 臂行 + G 族门 + `proj_share_u6` 不可比警告）、40 → **44 坑**（新增 41 端点量构造性饱和 / 42 归一化方向与仪器伪影 / 43 换族只能用秩 / 44 legacy payload 非数字键）；`rdc-phase-closeout` 10 → **12 条**教训（新增 11 append-only 必须配"按字节回滚"脚本 / 12 自检锚点须先对追加源文件预检）。
- **MEMORY 压缩重写**：16,847 → 14,231 → **%d B（%d 字符）**，解决注入截断问题；Ledger n=**295**、Phase 12 条目、铁律 **(r)(s)** 入册；§7 死线改为 **Phase 13 = 位点间配对 bootstrap**。
""" % (len(mem), len(mem.decode('utf-8')))

t1 = b0.decode('utf-8').rstrip('\r\n') + '\n\n' + sec.strip('\n') + '\n'
open(WLOG, 'wb').write(t1.encode('utf-8'))
b1 = open(WLOG, 'rb').read()
print('wlog bytes %d -> %d (+%d) ; lines %d' % (len(b0), len(b1), len(b1) - len(b0), len(b1.split(b'\n'))))
print('wlog sha256', hashlib.sha256(b1).hexdigest())
t = b1.decode('utf-8')
for k in ['49 个分区 / 0 FAIL', '0.000e+00', '5.329e-15', '端点差', '00:14:50', '13 臂', '44 坑', '12 条', '295', '(r)(s)']:
    print('  anchor %r count=%d' % (k, t.count(k)))
