# -*- coding: utf-8 -*-
"""Phase 13 收尾链末环：给 rdc-phase-closeout 技能补第 15 条教训。

教训来源（本轮实证）：A8 勘误触发「MEMO / Ledger / wlog 三链按基线 sha256 回滚 →
重跑 whole chain」后，产物正确但 disk_verify 的 [H_docs] 仍硬编码回滚前的期望常量
⇒ 3 个伪 FAIL。纪律：verify 只硬编码当前基线值；回滚重跑后必须重跑 verify。

纪律：逐处 assert count==1 + 回读复核（铁律 o）。
"""
import io
import os
import hashlib
import py_compile

P = r'C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md'
src = io.open(P, encoding='utf-8').read()
sha_before = hashlib.sha256(src.encode('utf-8')).hexdigest()

HDR_OLD = '**Phase 8–13 实测的 14 条收尾教训**：'
HDR_NEW = '**Phase 8–13 实测的 15 条收尾教训**：'

ANCHOR = '## 参照实现（Phase 3125，210/210 全绿；Phase 3128，19/19 全绿）'

LESSON = (
    '15. **勘误触发「三链回滚 + 重跑」后，必须同步更新复核脚本里的硬编码期望常量（Phase 13 实证）**：'
    'Phase 13 的 A8 事实勘误（`XH_RANGE` **并未**跌破 G0 的 0.10 阈值，只是裕度由 9.39% 压到 0.73%）'
    '按教训 11 的路径把 **MEMO / Ledger / wlog 三链按基线 sha256 回滚、再重跑 whole chain**，'
    '产物本身完全正确；但 `disk_verify_phase13.py` 的 `[H_docs]` 分区仍硬编码回滚**前**的期望值'
    '（`MEMO bytes == 308702`、`MEMO sha8 == 0c192b97`、`baseline.bytes == 308702`），'
    '实际已是 `308863` / `4f9b4574` ⇒ **同一份正确产物被判成 3 个伪 FAIL**。'
    '⇒ 纪律三条：① 复核脚本**只硬编码「当前」基线值**，且这些常量必须与 `closeout_docs` 刷新的 '
    '`memo_baseline.json` **同源**；② **任何回滚重跑之后，收尾链末尾的独立磁盘复核必须整脚本重跑一遍**'
    '（本轮 `90 checks / 0 FAIL`），不得沿用回滚前的那份报告；③ 把「复核脚本常量过期」'
    '**列为回滚方案的已知副作用**写进 `rollback_*.py` / `correct_*.py` 的注释里。'
    '修法仍走 Python 补丁（逐处 `assert count==1` + 回读 + `py_compile`），**只改期望常量、不动判据逻辑**；'
    '本轮旧值残留自检 `count(308702)==0`、`count(0c192b97)==0` 均为真。\n'
)

assert src.count(HDR_OLD) == 1, 'HDR count=%d' % src.count(HDR_OLD)
assert src.count(ANCHOR) == 1, 'ANCHOR count=%d' % src.count(ANCHOR)

src = src.replace(HDR_OLD, HDR_NEW)
src = src.replace(ANCHOR, LESSON + '\n' + ANCHOR)

io.open(P, 'w', encoding='utf-8', newline='').write(src)

# ---- 回读复核 ----
back = io.open(P, encoding='utf-8').read()
assert back.count(HDR_NEW) == 1 and back.count(HDR_OLD) == 0
assert back.count('15. **勘误触发') == 1
assert back.count('16. ') == 0
assert back.count(ANCHOR) == 1
# 教训编号连续 1..15
for i in range(1, 16):
    assert back.count('\n%d. **' % i) == 1, '教训 %d 计数异常' % i
sha_after = hashlib.sha256(back.encode('utf-8')).hexdigest()

print('[1] 头部 14 条 -> 15 条')
print('[2] 第 15 条已插入（锚点前），编号 1..15 各出现 1 次')
print('    bytes %d -> %d' % (len(sha_before) and len(io.open(P, encoding="utf-8").read()), len(back)))
print('    sha256 %s -> %s' % (sha_before[:8], sha_after[:8]))
print('    readback OK')
