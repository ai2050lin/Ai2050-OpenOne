# -*- coding: utf-8 -*-
"""Append Phase 2935 section to AGI_GPT5_MEMO.md."""
import hashlib

P = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
REP = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2935_memo_append_report.txt')

sec = """## Phase 2935: null 放大机制解剖——null tid 重采样分离上下文统计效应与 token 身份效应 [2026-09-19 17:07]

### 目的与设计（预注册，execution.json 先落盘）
- 问题：2934 发现 null 上下文 CI 全面放大（LOAD ×1.83、DEEP ×2.00）——放大由 null tid 特定身份驱动（H1）还是由上下文缺乏语义约束驱动（H2）。
- 设计：func + 4 组独立重采样 null tid 集（seed 2896 原组/2914/2915/2916，2927 采样规则 verbatim），pos1 真实消融 × 1120 门控格 × 5 条件 = 5600 前向；2934 修正协议 verbatim（per-condition batch57）。
- 判决映射（冻结）：P1 四组全复现律 且 P2/P3 中位 Spearman ≥0.9 => null_amp_context_general；四组全复现 且 ≥0.5 => null_amp_mixed；否则 null_amp_token_unstable。

### 运行记录
- run1 崩溃于 P2 序列化段（cell tuple 索引 1-D 数组 IndexError，计算已全部完成、判决未受影响）；修 pos_of 映射后按纪律删产物重跑。
- run2（267 s）锚 7/7：a1 2.17e-08（**第十二次连续前向锚定**）；a2 0.0；a4 469/382/295；a5 185.70；**a6 func CI_rel vs 2933 全 1120 格 diff = 0.00e+00（bit 级）**；a7 func 分离 185.7 > 全部 null 组（77.3/106.1/106.1/81.5）。

### 结果（冻结判决 null_amp_mixed）
- **P1 律复制：4 组 null 全部稳定**——rho −0.6574/−0.6608/−0.6619/−0.6420（p_band≤6e-4）——lin_r-CI 律不依赖任何特定 null token 抽样。
- **P2 放大比稳定性：pairwise 中位 0.8044**（6 对 0.796-0.818）——稳定但低于 0.9 的"上下文普适"门槛。
- **P3 原始 CI 一致性：中位 0.9420**——CI 剖面跨 null 重采样高度一致。
- **判决 null_amp_mixed**：放大主体是上下文统计效应（H2），含少数 token 身份成分（per-cell 跨组 std 中位 0.24，相对 amp ~1.9 约 13%）。

### seal 取证（三项新发现）
1. **放大梯度反转**：放大最多的恰是承重带/早层（L6 ×3.15、L3 ×2.82、L1 ×2.51），深层最小（L11 ×1.59）——**放大比与绝对 CI 跨层反相关**：func 语义上下文在 CI 最大的地方抑制最强（语义约束把词位读出"锚定"，消融不敏感处本就无需锚定）。
2. **幸存核二分**：L6 型幸存者高放大（amp 2.56-3.18）vs L8/L9 型幸存者几乎不放大（amp 1.20-1.26，func CI 已高）——幸存核两类成员对上下文统计的依赖模式不同。
3. 骨架格略不易放大（1.77 vs 其余 1.91）；极端 token 身份离群格存在（(14,6) 跨组 amp std 1.38，均值 7.14）。

### 硬伤
- 4 组重采样仍属小 R（放大比置信区间宽）；token 身份成分未归因（词频/嵌入范数未测）；放大比的分母（func CI）含自身测量噪声。
- run1 序列化 IndexError 属实现级（非判据级），如实登记。

### 文件与 SHA256-8
- 脚本 tests/glm5/phase2935_null_amp_anatomy.py: 29f2ea61
- execution.json: 9a11de15（created 2026-09-19T17:07:10）
- result.json: 4e62c6fc（final_verdict=null_amp_mixed，runtime 266.8 s）
- null_amp_anatomy.npz: 3b947b5d（cells/ci_rel 5x1120/amp 4x1120/s_base/scale/sep/dirs_word）
- 源：2887 e4835a87；2927 84fec594；2929 57ed5651；2930 cb655825；2931 5307afe1；2933 1ff6df21
- 产物目录 tests/glm5/result/rdc_query_construction_20260913/phase2935/null_amp_anatomy/
- Ledger：M2935_null_amp_anatomy 入账，measurements 73->74，L14 connects 41->42，ledger sha256-8 = 57a06980

### 接续（2936 候选）
- A（主选）：**语义上下文抑制律**——放大比与层内 lin_r/CI 的关系形式化（L1-L6 放大 2.5-3.2 vs L8-L28 1.6-1.7 的剖面建模），检验"锚定假说"：语义上下文抑制量 ∝ 该层 CI 绝对量（零新前向，2935 npz 复用 + 2934 npz CI）。
- B：eps 扫描线性度（lin_r ∈ {0.1, 0.3, 1.0} 偶阶 ~eps 缩放，一次前向族）。
- C（零前向）：h4 L1<->L19 复用子空间主角度（roadmap 遗留项）。
- D：承重带跨模型复现（glm4 双条件消融子采样，一次前向；只主张 lin_r<0.9 层）。
"""

h = hashlib.sha256()
h.update(open(P, 'rb').read())
before = h.hexdigest()[:8]
with open(P, 'a', encoding='utf-8') as f:
    f.write('\n' + sec)
lines = open(P, encoding='utf-8').read().splitlines()
title_ok = any(l.startswith('## Phase 2935:') for l in lines)
h2 = hashlib.sha256()
h2.update(open(P, 'rb').read())
rep = ['before_sha=%s' % before,
       'after_sha=%s' % h2.hexdigest()[:8],
       'total_lines=%d' % len(lines),
       'title_ok=%s' % title_ok,
       'tail=%s' % lines[-1][:80]]
open(REP, 'w', encoding='utf-8').write('\n'.join(rep) + '\n')
print('memo ok')
