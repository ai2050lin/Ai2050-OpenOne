# -*- coding: utf-8 -*-
"""Phase 14 收尾链补充：把「独立磁盘复核首次跑出的 4 处断言缺陷」追加进当日 wlog。
append-only；只 append，不动既有字节（前缀锚核对）。
"""
import io
import os
import hashlib

W = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\2026-10-02.md'
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\Phase14\wlog_supplement_phase14.txt'

b0 = open(W, 'rb').read()
PRE = len(b0)
SEC = """
## Phase 14 收尾链补充：独立磁盘复核首次跑出 4 处**断言缺陷**（115 checks / FAIL = 0，修正后）

- **性质**：全部是本轮新写的 `disk_verify_phase14.py` **自身的缺陷**（2 处判据错配 + 2 处常量错），**产物本身无问题**；修正后 115 checks / FAIL = 0，且两坐标 jumps 与置换 null 均 `0.0e+00` 逐位。
- **真缺陷 1（技术陷阱，值得入册）**：null 重放用的剖面我从 `A1_curves` 的 `y`（`curve_from_rows` 产出）取，而主脚本 `full_concentration` 实际走 `PM.mean(axis=2)/FULL_SWAP`。**两条路径的浮点求和顺序不同** ⇒ 18 位点里 8 个 `xhalf` 差 1 ULP（jumps `max|d| = 3.331e-16`）⇒ 置换 null 95 分位无法逐位复现（A1 `4.4e-16` / A8 `2.6e-15`）。改走 PM 路径后 **A1/A8、x/J 四个 jumps 与两个 null 全部 `0.0e+00`**。
  ⇒ **新坑**：「同 seed 逐位复现」必须连**浮点求和顺序**一起复刻；独立复核不能只校「统计量公式」，还要校「剖面的构造路径」。
- **真缺陷 2（口径错配）**：我把 A1 的 `alpha=1 == FULL_SWAP_pairs` **逐对恒等式**（`F30a`，设计上**只覆盖 A1**，见主脚本 L685 `for s in _A1_SITES`）误套到 A8。实测 A8 的 `alpha=1` **不是满替换**：端点 `y(i,1)` 逐位等于 Phase 12 `recover(site_i)`（**18/18，`max|d| = 0.000e+00`**），且 18/18 支撑的 `alpha=1` 逐对向量**都不等于** `FULL_SWAP_pairs`。
  ⇒ 改为「A8 口径分离」正向核验，并由此**新增一条逐位锚**：A8 端点 ≡ Phase 12 `recover`（与 F29 面板级一致，本轮补到逐位点级）。
- **另两处常量错**：`F30a.n_pairs` 硬编码 24，实为 **432**（= 18 位点 × 24 对）；`P1..P7 全 PASS` 与预注册事实相反，应核**确切模式**（P1/P2/P3/P7 真，P4/P5/P6 假）。
- **修正方式**：`patch_disk_verify_phase14.py`（7 处 `assert count==1` + 回读 + `py_compile`）；残旧断言自检 3 项（`P1..P7 全 PASS` / `n_pairs == 24` / `A8 全 18 支撑 alpha=1 逐对`）残留均为 `0`。
- **再入册一条收尾教训（第 19 条）**：**独立复核脚本必须自己先跑通再进收尾链**——本轮 verify 脚本从未被执行过就直接进了链（首跑崩在 `NameError: AM1`，修完才暴露 6 个 FAIL、再修出真缺陷 2）。⇒ 复核脚本写完**立刻空跑一遍**，把「脚本自身缺陷」与「产物缺陷」在判决前分离。
- **技能同步（本轮两处计数已过期，已修）**：`rdc-main-axis-probe` §4 标题 `46 条` → **49 条**（条目实际已有 49）；`rdc-phase-closeout` §N 线标题 `Phase 8 → Phase 12` → **Phase 8 → Phase 14**、`已跑通六次` → **七次**、`15 条收尾教训` → **18 条**（并随本轮补第 19 条）。
"""
t1 = b0.decode('utf-8').rstrip('\r\n') + '\n\n' + SEC.strip('\n') + '\n'
open(W, 'wb').write(t1.encode('utf-8'))
b1 = open(W, 'rb').read()

chk = [
    ('append-only：前缀 %d B 逐字节未变' % PRE, hashlib.sha256(b1[:PRE]).hexdigest() == hashlib.sha256(b0[:PRE]).hexdigest()),
    ('bytes %d -> %d' % (len(b0), len(b1)), len(b1) > len(b0)),
    ('Phase 14 补充节存在', t1.count('## Phase 14 收尾链补充') == 1),
    ('无新增裸 LF', (b1.count(b'\n') - b1.count(b'\r\n')) == (b0.count(b'\n') - b0.count(b'\r\n'))),
]
L = ['=== wlog_supplement_phase14 ===']
for k, v in chk:
    L.append('  %-40s %s' % (k, 'OK' if v else '**FAIL**'))
L.append('sha256 = ' + hashlib.sha256(b1).hexdigest())
L.append('ALL OK' if all(v for _, v in chk) else 'HAS FAIL')
io.open(OUT, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('\n'.join(L))
