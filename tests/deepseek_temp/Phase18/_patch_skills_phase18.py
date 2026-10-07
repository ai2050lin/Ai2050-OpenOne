# -*- coding: utf-8 -*-
"""Phase 18 技能同步：rdc-main-axis-probe 坑 57→58；rdc-phase-closeout 教训 28→29。"""
import io
import os

SK = r'C:\Users\Admin\.workbuddy\skills'

PIT58 = (
    "\n58. **比值型诊断量在「近零分母」处会被放大 ⇒ 峰值/次大之比不可作硬门；"
    "且阈值的**标定支撑域**必须与实现的**计算支撑域**逐字一致（Phase 18 实证：一条预注册预测因此被判 FAIL，"
    "而其中的位置子命题仍成立）**：Phase 18 用 `r_lin,ℓ = |b_all − (b_mlp + b_attn)| / max(|b_all|, eps)` "
    "做「层内超可加性」诊断量，seal 的 P6 判据是「`argmax r_lin == L*_own` **且** 峰值/次大 ≥ 3」。\n"
    "    - **实况**：`argmax r_lin` 在三臂都等于该臂写入窗 `L*_own`（**位置子命题成立**）；"
    "但**峰值比**在实现所用的 `ALL_SITES = 1..L−2` 支撑上只有 **1.55**（次大在 **L2**）——"
    "因为浅端 `|b_all|` 极小，比值被分母放大。\n"
    "    - **根因**：seal 的 `predictions.P6.rationale` 引用的「次大 0.186（L26）、比值 4.05」"
    "取自**更早的 REACH 受限网格**（那一轮 `SITES ⊆ REACH`），而生产实现跑在**全支撑**上；"
    "两项支撑给出完全不同的「次大」与比值（REACH 受限下为 3.00，次大在 L23）。\n"
    "    - **纪律（三条）**：① 凡 **seal 里引用探针数字**的地方，必须写明该数字的**支撑域**，"
    "并断言其与实现的支撑域**逐字一致**（承接坑 57「口径歧义靠独立实现交叉验证捕捉」——本轮由"
    "「探针**全支撑** vs 生产」的逐位比对检出）；② **比值型量**（`|residual|/|ref|`）不得作硬门 —— "
    "改用**绝对残差**或给分母设下限，否则「峰/次峰分离度」会被近零分母污染；③ 一旦发现是**阈值标定** "
    "而非**主量**错了，**不追改判据**：按 seal 字面判 FAIL，同时把**子命题**（`argmax` 位置）**单独报告**，"
    "并在 MEMO 勘误节写清支撑域差异与两支撑下的数值。\n"
)


LESSON29 = (
    "\n29. **预注册引用的探针证据必须带「支撑域」标注 —— 阈值取自受限支撑、实现跑在全支撑 ⇒ "
    "判据被近零分母污染（Phase 18 实证：P6 峰值比 rationale 4.05 vs 实现 1.55，一条预测 FAIL，"
    "但其中的位置子命题仍成立）**：\n"
    "    - **(a) 症状**：`r_lin = |b_all − (b_mlp+b_attn)| / |b_all|` 的峰值比在 seal 里写 ≥3（rationale 引 4.05），"
    "生产实测 **1.55**（A0）；`argmax` 却仍等于 `L*_own`。一条预测 FAIL，但**不是因为机制错**，"
    "而是因为**阈值标定用的支撑域 ≠ 实现用的支撑域**。\n"
    "    - **(b) 处置模板**：① **不改判据**（预注册纪律）⇒ 按字面记 FAIL；② 把**子命题**"
    "（`argmax == L*_own`）单独报告为「成立」；③ MEMO 同轮勘误节新增 `E-rlin`，写清两支撑下的数值"
    "（`ALL_SITES` 1.55 / `REACH` 3.00）与「4.05 在两种支撑上都不复现」；④ 后继若继续用该量，"
    "改用绝对残差或加分母下限。\n"
    "    - **(c) 检出手段**：**探针（全支撑）vs 生产**的**逐位比对**——把探针的 `rlin` 全谱与生产的 "
    "`rlin_by_site` 对齐，一眼看出次大位点从 L26 变成 L2。**纪律**：seal 里凡引用探针数字，"
    "都必须同时写明**支撑域**（承接教训 27「复核脚本自身也会错」的脸谱：这里错的是**seal 的证据引用**）。\n"
    "    - **(d) 复用**：`disk_verify_*` 增加一条 —— 「seal 引用的探针数字 == 该数字在**实现支撑域**上的重算」，"
    "不成立就报 FAIL（本轮即由该思路定位）。\n"
)


def patch(fn, anchor, add, label):
    p = os.path.join(SK, fn)
    s = io.open(p, encoding='utf-8').read()
    a = s.count(anchor)
    assert a == 1, '%s anchor count=%d' % (fn, a)
    s = s.replace(anchor, add + anchor)
    io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
    c = io.open(p, encoding='utf-8').read()
    print('%-28s patched (%s) ; new anchor count=%d' % (fn, label, c.count(anchor)))


patch('rdc-main-axis-probe\\SKILL.md', '\n## 5 代码骨架要点', PIT58, 'pit 58')
patch('rdc-phase-closeout\\SKILL.md', '\n## 参照实现（Phase 3125', LESSON29, 'lesson 29')

# 复核编号存在
m = io.open(os.path.join(SK, 'rdc-main-axis-probe', 'SKILL.md'), encoding='utf-8').read()
c = io.open(os.path.join(SK, 'rdc-phase-closeout', 'SKILL.md'), encoding='utf-8').read()
print('pit 58 present:', '58. **比值型诊断量' in m)
print('lesson 29 present:', '29. **预注册引用的探针证据' in c)
