# -*- coding: utf-8 -*-
"""Phase 14 收尾第二阶段补丁：生成器 CRLF + 两个技能的过期计数与新增条目。

（1）`closeout_docs_phase14.py`：wlog 写入改 CRLF（原用 '\\n' ⇒ 混合 EOL，复核实证）。
（2）`rdc-main-axis-probe`：§4 标题 46 条 → 50 条；新增坑 50（独立复核必须复刻剖面构造路径）。
（3）`rdc-phase-closeout`：§N 线标题 Phase 8→12 改 Phase 8→14；六次 → 七次；15 条 → 20 条；
     新增教训 19（复核脚本必须先空跑）、20（wlog EOL 由生成器决定 + 修复范围）。
逐处 assert count==1 + 回读 + py_compile。
"""
import io
import os
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
COD = os.path.join(ROOT, 'tests', 'deepseek', 'Phase14', 'closeout_docs_phase14.py')
SK1 = r'C:\Users\Admin\.workbuddy\skills\rdc-main-axis-probe\SKILL.md'
SK2 = r'C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md'
OUT = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase14', 'patch_skills_phase14b.txt')

PIT50 = (
    "50. **独立复核必须复刻「剖面的构造路径」，只复刻统计量公式会差 1 ULP（Phase 14 复核实证）**："
    "同一份 `result` 里位点剖面有两条**不同**的浮点路径 —— ① `curve_from_rows` 的 `A1_curves[s].y`"
    "（按 pair 平均 `dDonor` 再统一除 `FULL_SWAP`）；② `full_concentration` 的 `PM.mean(axis=2)/FULL_SWAP`"
    "（在 `(nS, nα, nP)` 张量上沿 pair 轴平均）。数学等价但**求和顺序不同** ⇒ 18 位点里 **8 个** `xhalf` 差 "
    "**1 ULP**（`jumps max|d| = 3.331e-16`）⇒ 依赖 jumps 的**置换零假设 95 分位**无法逐位复现"
    "（实测 `4.4e-16` / `2.6e-15`，其余全部 `0.0e+00`）；改走 ② 后 **A1/A8 × x/J 四个 jumps 与两个 null 全部 `0.0e+00`**。"
    "**对策**：独立复核脚本一律用**与主脚本同一条张量路径**重建剖面，并在报告里给**三级自检**"
    "（点估计 `max|d|` / jumps `max|d|` / null 95 分位 `max|d|`），三级全 `0.0e+00` 才算逐位复现。"
    "**并且**：`α=1` 的语义**逐臂不同** —— A1（全位置替换）`α=1` 是真满替换（逐对 == `FULL_SWAP_pairs`，元素级，`n = 18×24 = 432`）；"
    "A8（逐层累积支撑）`α=1` **不是**满替换，其端点 `y(i,1)` 逐位等于上一 Phase 的 `recover(site_i)`"
    "（**18/18，`max|d| = 0.000e+00`**），18/18 支撑的逐对向量都**不**等于 `FULL_SWAP_pairs` ⇒ **恒等式断言不得跨臂套用**。"
    "\r\n\r\n")

L19 = (
    "19. **独立复核脚本必须「先空跑」再进收尾链 —— 否则脚本自身缺陷会被当成产物缺陷（Phase 14 实证）**："
    "`disk_verify_phase14.py` 写完**从未执行**就进了链：首跑直接崩在 `NameError: AM1`（只定义了路径 `AM1P`、忘了 `json.load`），"
    "修完才暴露 **6 个 FAIL** —— 其中 **2 个是判据错配**（把 A1 的 `α=1 == FULL_SWAP_pairs` 逐对恒等式误套到 A8；该恒等式在设计上只覆盖 A1）、"
    "**2 个是常量错**（`F30a.n_pairs` 实为 **432** 而非 24；`P1..P7 全 PASS` 与预注册事实相反，该 Phase 恰有 P4/P5/P6 判 FAIL）、"
    "**2 个是真技术陷阱**（见 `rdc-main-axis-probe` 坑 50：剖面构造路径的浮点求和顺序）。"
    "**对策**：① 复核脚本写完**立刻空跑一遍**，把「脚本崩溃 / 断言写错」在**判决之前**修净；"
    "② 凡引用 `result` / `exec` / `Ledger` 的键，**先用探针确认键名与类型**（本轮 `E14['bootstrap']` 是 `BS`/`BP` 而非 `B`/`B_perm`）；"
    "③ 断言里**不得写与预注册事实相反的期望**；④ 补丁脚本自身的自检也要写对 —— 若新文本**有意保留**旧行，就不能再断言「旧文本已消失」（本轮踩过）。"
    "\r\n\r\n")

L20 = (
    "20. **append-only 日志（wlog）的 EOL 由生成器决定：生成器必须写 CRLF；EOL 修复只允许在「内容全部由本轮写入」的文件上做（Phase 14 实证）**："
    "`closeout_docs` 与 wlog 补充脚本都用 `'\\n'` 拼接 ⇒ `2026-10-02.md` 变成**混合 EOL**（`lf 99 / crlf 10`），"
    "而本项目 **11/14** 个 wlog 是 CRLF-only（`09-15..09-26 / 09-28..09-30`）。"
    "**对策**：① 生成器统一写 `'\\r\\n'`（本轮已改 `closeout_docs_phase14.py`）；"
    "② 若某 wlog 已是混合、且其内容**全部由本轮写入**，可做**一次性规范化** —— 按**文本行**重排 EOL 并断言**行列表逐条相同**"
    "（信息零变化；本轮 `29103 → 29192 B`，`+89` 恰等于原裸 LF 行数 `99 − 10`，独立复核 9/9 PASS）；"
    "③ 历史文件（`2026-09-18` / `2026-09-27` / `2026-10-01`）**不动**；"
    "④ EOL 改动会让此前记录里的 wlog `sha256` 过期 ⇒ 必须**单出一份** `wlog_eol_repair_*.txt/.json` 记录 before/after 与副作用；"
    "⑤ 复核脚本**不要**对 wlog 断言 `bare_lf == 0` —— 那是 **MEMO 专属**纪律。"
    "\r\n\r\n")

TASKS = []

# ---------- (1) closeout_docs：wlog 写 CRLF ----------
TASKS.append((COD, (
    "t1 = t0.rstrip('\\r\\n') + '\\n\\n' + sec.strip('\\n') + '\\n'\nopen(WLOG, 'wb').write(t1.encode('utf-8'))"
), (
    "# wlog 与 MEMO 同惯例：CRLF（Phase 14 复核发现本行原用 '\\n' ⇒ 混合 EOL，lf 99 / crlf 10）\n"
    "_sec_n = sec.replace('\\r\\n', '\\n').replace('\\n', '\\r\\n').strip('\\r\\n')\n"
    "t1 = t0.rstrip('\\r\\n') + '\\r\\n\\r\\n' + _sec_n + '\\r\\n'\nopen(WLOG, 'wb').write(t1.encode('utf-8'))"
)))

# ---------- (2) rdc-main-axis-probe ----------
TASKS.append((SK1, "## 4 已实测的坑（46 条，逐条对应数值）", "## 4 已实测的坑（50 条，逐条对应数值）"))
TASKS.append((SK1, "\r\n\r\n## 5 代码骨架要点", "\r\n\r\n" + PIT50 + "## 5 代码骨架要点"))

# ---------- (3) rdc-phase-closeout ----------
TASKS.append((SK2, "## N 线（deepseek）Phase 收尾实证（Phase 8 → Phase 12，2026-10-01/02）",
              "## N 线（deepseek）Phase 收尾实证（Phase 8 → Phase 14，2026-10-01/02）"))
TASKS.append((SK2, "这条线的收尾链已跑通六次（Phase 8/9/10/11/12/13）",
              "这条线的收尾链已跑通七次（Phase 8/9/10/11/12/13/14）"))
TASKS.append((SK2, "**Phase 8–13 实测的 15 条收尾教训**：", "**Phase 8–14 实测的 20 条收尾教训**："))
TASKS.append((SK2, "\r\n## 参照实现（Phase 3125", "\r\n" + L19 + L20 + "## 参照实现（Phase 3125"))

L = ['=== patch_skills_phase14b ===', '']
buf = {}
for path, old, new in TASKS:
    if path not in buf:
        buf[path] = open(path, 'rb').read().decode('utf-8')
    c = buf[path].count(old)
    ok = 'OK' if c == 1 else '**FAIL**'
    L.append('  [%s] count=%d %s  :: %s' % (os.path.basename(path), c, ok, old[:56].replace('\r', '\\r').replace('\n', '\\n')))
    assert c == 1, 'FAIL count=%d for %r' % (c, old[:80])
    buf[path] = buf[path].replace(old, new)

for path, t in buf.items():
    b0 = open(path, 'rb').read()
    open(path, 'wb').write(t.encode('utf-8'))
    b1 = open(path, 'rb').read()
    L.append('')
    L.append('  %s : %d -> %d (%+d) ; sha8 %s -> %s' % (os.path.basename(path), len(b0), len(b1), len(b1) - len(b0),
                                                       hashlib.sha256(b0).hexdigest()[:8], hashlib.sha256(b1).hexdigest()[:8]))

# 回读复核
r1 = io.open(SK1, encoding='utf-8').read()
r2 = io.open(SK2, encoding='utf-8').read()
r0 = io.open(COD, encoding='utf-8').read()
L += ['', '--- 回读复核 ---']
L.append('  SK1 标题 50 条 : %s' % ('OK' if r1.count('## 4 已实测的坑（50 条，逐条对应数值）') == 1 else '**FAIL**'))
L.append('  SK1 坑 50 存在  : %s' % ('OK' if r1.count('50. **独立复核必须复刻') == 1 else '**FAIL**'))
L.append('  SK1 旧 46 条残留 : %d (须 0)' % r1.count('（46 条，逐条对应数值）'))
L.append('  SK2 Phase 8→14 : %s' % ('OK' if r2.count('Phase 收尾实证（Phase 8 → Phase 14') == 1 else '**FAIL**'))
L.append('  SK2 七次       : %s' % ('OK' if r2.count('已跑通七次（Phase 8/9/10/11/12/13/14）') == 1 else '**FAIL**'))
L.append('  SK2 20 条教训  : %s' % ('OK' if r2.count('**Phase 8–14 实测的 20 条收尾教训**：') == 1 else '**FAIL**'))
L.append('  SK2 教训 19/20 : %s' % ('OK' if (r2.count('19. **独立复核脚本必须') == 1 and r2.count('20. **append-only 日志') == 1) else '**FAIL**'))
L.append('  SK2 旧标题残留 : %d / %d / %d (须全 0)' % (r2.count('Phase 8 → Phase 12'), r2.count('已跑通六次'), r2.count('15 条收尾教训')))
L.append('  COD wlog CRLF  : %s' % ('OK' if r0.count("+ '\\r\\n\\r\\n' + _sec_n + '\\r\\n'") == 1 else '**FAIL**'))
ALL = (r1.count('## 4 已实测的坑（50 条，逐条对应数值）') == 1 and r1.count('50. **独立复核必须复刻') == 1
       and r1.count('（46 条，逐条对应数值）') == 0
       and r2.count('Phase 收尾实证（Phase 8 → Phase 14') == 1 and r2.count('已跑通七次（Phase 8/9/10/11/12/13/14）') == 1
       and r2.count('**Phase 8–14 实测的 20 条收尾教训**：') == 1 and r2.count('19. **独立复核脚本必须') == 1
       and r2.count('20. **append-only 日志') == 1 and r2.count('Phase 8 → Phase 12') == 0
       and r2.count('已跑通六次') == 0 and r2.count('15 条收尾教训') == 0
       and r0.count("+ '\\r\\n\\r\\n' + _sec_n + '\\r\\n'") == 1)
L += ['', 'ALL OK' if ALL else 'HAS FAIL']
io.open(OUT, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('\n'.join(L))
assert ALL
