# -*- coding: utf-8 -*-
"""Append R1 review entry to AGI_DEEPSEEK_MEMO.md (CRLF preserved, append-only)."""
import hashlib

P = r'D:\AI2050\Ai2050-OpenOne\research\deepseek\docs\AGI_DEEPSEEK_MEMO.md'

before_sha = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]

entry = """
## 复核 R1: Phase 1-7（N1-N3 线）测试思路与结果复核（非 Phase 编号）[2026-10-01 20:52]

> 类型：**复核（audit）**，不占 Phase 编号。全文报告：`tests/gpt5_temp/memo_review_20261001/REVIEW_REPORT.md`。
> 方法：全文精读 1448 行 + 盘上产物核验（33/33 存在、sha8 抽验）+ 原始报告关键数字抽查约 30 处 + 三份 design seal 对照。

### 裁决

1. **数字可信**：抽查约 30 处关键数字与原始报告**零不一致**；e2 脚本 sha8 af449ff9 / 报告 d6de235f 与记载一致；三个 seal 均在观测前冻结落盘。
2. **思路总体正确**：三次自我降级（N1c 重建窗→N2 否证；N2c 承诺曲线→N2h1 置换分解；jump>3.0→jump/max）构成健康纠错链。
3. **核心发现成立**（各自证据范围内）：is-a 分布化重构（share_max 3.0%）；5 维类别子空间充分必要（B/A 0.994-1.010、C_resid/D_rand≈0）；族间子空间近正交（0.002-0.025）；写入窗族特异（相对深度 0.03-0.17 vs 0.60-0.92）。

### P1 级纠错（必须执行）

**R1-P1：N3 K_d 极性判据被本 memo Phase 7 头条改写（seal 漂移）。**
- Seal 冻结：K_d = 否定模板下 B_cat 与 A_full 同号且 B/A≥0.70 → 解耦；**否则判「极性-内容纠缠」**。
- 实测（报告原文）：qwen3-4b B/A=0.526、glm4 B/A=0.333（报告总结行均为「跨极性=FAIL」）；qwen2.5 B/A=7.380 系 A_full=+0.012 近地板的退化比值，无意义。
- Phase 7 头条"极性撤轴≠关通道（3/3）……内容与极性可分离的第一个定量证据"与冻结判据的否证分支**正面冲突**。"跨上下文注入仍有效"（+0.398~+1.945）是 K_d 未覆盖的 post-hoc 观察，只可作假说。
- 处置：① 本条即纠错记录（append-only，不改原文）；② "撤轴≠关通道"降级为 post-hoc 假设 P-N3b，按其自写否证条件（肯定/否定各自构造子空间主角谱 cos<0.3）预注册后复测；③ 修复前"内容与路由可分离"不得进入 RDC 已确立拼图；④ P-N3b 复测前须先修否定臂行为基线（F3neg base 分数 7.139 vs 肯定 7.739，行为上根本不是否定任务）。

### P2 级修正（表述降级）

- **R1-P2 维数定律部分同义反复**：U 定义为 6 个去均值类均值 SVD（秩≤5），rank sweep 到 k=5 时按构造恒等包含全部类间张成 → "6/6 在 k=G-1 饱和"是估计量数学必然，非模型性质。有经验内容的是**谱形状**（rank-1 仅 11-18%、逐级上升=类均值一般位置展开，无低秩坍缩）。"新增一类只需 +1 方向"降级：G 类线性可分码下界即 G-1（分类数学必然）。
- **R1-P3 零化必要性对照维度失配**：随机 5 维分量仅携带 ~5/d 状态能量（d=2560），170-6000× 主要由维度比制造。真实必要性证据 = 绝对掉分 5.9-6.8（base~+10）+ C_resid≈0。补对照：范数匹配随机 5 维、top-5 主成分（高方差非类）零化。
- **R1-P4 E_same 对照偏弱**：同类供体 P_U Δ 本身近零（构造使然）；补"同类跨上下文供体"臂。
- **R1-P5 绑定规则**："6/6"实为 5/5 可用 + 1 排除（gemma）；untied 仅 n=2 且与家族/规模混杂 → 补 qwen3-14b（untied）第三点 + 14B 规模缺口。

### P3 级挂账

- R1-P6：K4（E2 死线：20 paraphrase × 12 多义词 × 3 模型 + R_ℓ 零带）悬置未判 → 排期或正式改判挂起。
- R1-P7：N1 A 臂逐词峰层 苹果/小米/病毒@L16、杜鹃@L10 → E2 的"L10-L12 窗"实为 L10-L16 宽窗，须回写。
- R1-P8：D_rand 单点、无 5-seed 噪声带（N2b 标准在 N2h1/N3 退化）。
- R1-P9：面板小（41/29 例、6 类、全单 token、全中文、全 cloze 末位）、无确认集（N3-δ 计划维持）。
- R1-P10：交互指数 I 表未落报告文件（仅 memo，数值 6.014/1.168=5.148 算术一致）；glm4 颜色族 L9-L20 负 A_full 挂账维持（N3-ε）。

### 优先级确认

N2h1-α（权重级归因，K_g 死线不变）> N3-δ 确认集（+P-N3b 并轮）> 对照补强（R1-P3/P4）> qwen3-14b 接入 > K4 处置。复核探针与产物：`tests/gpt5_temp/memo_review_20261001/`。
"""

with open(P, 'r', encoding='utf-8') as f:
    raw = f.read()

# detect line ending style
crlf = raw.count('\r\n')
lf_only = raw.count('\n') - crlf
style = '\r\n' if crlf >= lf_only else '\n'

if not raw.endswith(style):
    raw = raw + style

body = entry.replace('\n', style)
if not body.startswith(style):
    body = style + body.lstrip(style)

with open(P, 'a', encoding='utf-8', newline='') as f:
    f.write(body)

after_sha = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
print('before_sha8=%s after_sha8=%s style=%s appended_chars=%d' % (before_sha, after_sha, repr(style), len(body)))
