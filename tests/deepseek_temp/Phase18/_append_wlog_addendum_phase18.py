# -*- coding: utf-8 -*-
"""追加 Phase 18 收尾补记到当日 wlog（append-only、CRLF）。"""
import io, os, time, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
WLOG = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-02.md')
b0 = open(WLOG, 'rb').read()
t0 = b0.decode('utf-8')
MARK = '### Phase 18 收尾补记'
add = (
    '\r\n' + MARK + '（%s）\r\n\r\n'
    '- **收尾链全部完成**：Ledger n **301**（`ledger_sha256_8 = e2927674` → 刷新 `8e38b972`）；'
    'MEMO 435,628 → **460,387 B** / 4,103 → **4,363 行** / 17 → **18 标题**（P18@L4104，前缀逐字节未变、BOM/CRLF、`bare_lf 0`）；'
    '`_infra/memo_baseline.json` → `post-append-phase18`（history 13）。\r\n'
    '- **独立磁盘复核 216 项 0 FAIL**（独立重写区间求和 / `stat_com_layer` / `spearman` / 置换零假设；'
    '含 A0 的 REACH 域 = **4.053**、次大 **0.186@L26** 复现 seal rationale 的跨件核对）。\r\n'
    '- **本轮发现的 3 处「漂亮但错误」**：① 探针 `d_mlp ≡ 0`（行号写错，SMOKE 前捕获）；'
    '② **E-rlin 首稿**「4.05 在两种支撑上都不复现」**是错的** —— 实测 REACH 域逐位复现（4.053），'
    '错配只在「判据域 = `ALL_SITES`」；③ `disk_verify` 首版把「预期成功」写成断言 ⇒ **13 个假 FAIL**（已改为「重算→导出标签→比对」）。\r\n'
    '- **技能同步（含纠错）**：`rdc-main-axis-probe` 坑 57→**58**（比值型量近零分母 + 判据域/标定域一致 + 「先重算再落笔」+「单列判决是否改变」）；'
    '`rdc-phase-closeout` 教训 28→**29**（同题，并新增 (e) 勘误结论句先验证、(f) 独立复核不得写死预期）。\r\n'
    '- **判决**：`FID_ALL_PASS` / `ANCHOR_ALL_OK` / `BRIDGE_PARTIAL` / `MLP_DOMINANT_BEH_ALL` / `ATTRIBUTION_CONSISTENT_ALL` / '
    '`BEHAVIOR_SHALLOWER_ALL` / `EFFICACY_COUPLED_ALL` / `NO_WINDOW_CONTRAST_ALL` / `NULL_TAIL_HIGH_PARTIAL`；'
    '预测 **P1✓ P2✗ P3✓ P4✓ P5✓ P6✗ P7 描述性**。\r\n'
) % time.strftime('%H:%M')
if MARK in t0:
    print('already present -> skip')
else:
    t1 = t0.rstrip('\r\n') + add
    open(WLOG, 'wb').write(t1.encode('utf-8'))
    b1 = open(WLOG, 'rb').read()
    print('wlog bytes %d -> %d ; sha256 %s' % (len(b0), len(b1), hashlib.sha256(b1).hexdigest()[:16]))
    print('mark count =', b1.decode('utf-8').count(MARK))
