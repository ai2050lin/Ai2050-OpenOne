#!/usr/bin/env python3
"""Fix garbled Phase 236 text in AGI_GLM5_MEMO.md"""
import sys

memo_path = "research/glm5/docs/AGI_GLM5_MEMO.md"

# Read the file
with open(memo_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find Phase 236 start line
phase236_start = None
for i, line in enumerate(lines):
    if "Phase 236" in line:
        phase236_start = i
        break

if phase236_start is None:
    print("Phase 236 not found!")
    sys.exit(1)

print(f"Phase 236 starts at line {phase236_start + 1}")
print(f"Total lines before: {len(lines)}")

# Keep everything before Phase 236
before = lines[:phase236_start]

# New Phase 236 content (correct Chinese)
new_content = """## Phase 236: Program Geometry — Δ Structure Analysis [2026-05-19 23:45]

### 核心目标

从"现象描述"转向"数学结构验证"。Phase 235发现否定是加法机制且非对合,
但关键问题未解: Δ_not有多少个自由度? 是否有简洁的数学结构?

### 实验设计

- ExpA: Δ_not SVD自由度分析(最关键!) — 100-200句子对, SVD分解Δ矩阵
- ExpB: 跨控制算子Δ结构对比 — not/never/always/rarely/often
- ExpC: 长度控制双重否定 — 区分语义漂移vs方向位移vs长度效应

### 核心发现1: 否定维度跨模型极端分化(最重要的发现!)

| 模型 | HS k90 | LS k90 | HS top1% | LS top1% | 判定 |
|------|--------|--------|----------|----------|------|
| DS7B | 4 | 2 | 65.8% | 81.6% | 极低维 — 有简洁数学结构! |
| Qwen3 | 51 | 38 | 16.8% | 45.0% | 中高维 — 部分结构 |
| GLM4 | 60 | 62 | 17.0% | 13.7% | 高维 — 无统一结构 |

**DS7B的否定是极低维的!** logit空间仅需2个成分解释90%方差, top1=81.6%.
Qwen3次之(logit空间有45%集中度), GLM4完全没有集中结构.

### 核心发现2: DS7B的"否定瓶颈" — 中间层1维化

DS7B逐层SVD:
- L4: k90=89, top1=11.0% (高维, 否定还在计算)
- L8-L24: k90=1, top1=97.5%!!! (几乎完美1维!)
- L27: k90=14, top1=79.2% (最后层稍增)

Qwen3逐层SVD (对比):
- L6-L35: k90=47-59, top1=12-15% (全程高维, 无收敛)

**DS7B在中间层将否定坍缩到1维!** 这是极其重要的发现 — 说明DS7B找到了一种
极其简洁的否定表示方式, 而Qwen3/GLM4没有.

### 核心发现3: 双重否定的跨模型差异

| 模型 | KL(单否定) | KL(双否定) | KL(长度控制) | ratio | 判定 |
|------|-----------|-----------|-------------|-------|------|
| Qwen3 | 0.28 | 2.50 | 0.76 | 3.27 | 语义漂移主导(63%) |
| GLM4 | 0.60 | 1.35 | 1.39 | 0.97 | 长度效应主导 |
| DS7B | 0.27 | 0.65 | 0.75 | 0.87 | 长度效应主导 |

Qwen3: 双重否定的偏离3.3倍于等长肯定句 → 真实语义偏离
GLM4/DS7B: 双重否定的偏离约等于等长肯定句 → 主要是长度效应

**解释**: Qwen3的否定是语义重塑(高维), GLM4/DS7B的否定更接近局部修改(低维)
→ 双重否定对低维模型来说, 偏离主要来自句子变长而非语义漂移.

### 核心发现4: 跨算子共享子空间

所有5个算子(not/never/always/rarely/often)的子空间重叠0.6-0.8.
not/never最高重叠(共享否定子空间), not/always较低(不同语义).
但所有重叠都较高, 说明"同位置插入"的结构效应占主导.

跨算子k90:
- Qwen3: 所有算子k90=33-34, top1=17-19%
- GLM4: 所有算子k90=37-40, top1=15-20%
- DS7B: 所有算子k90=3-4, top1=62-67%

**所有算子的维度与否定类似** — 不是否定特有的, 而是算子类共享的维度特征.

### 对两份分析的综合评判

分析一(几何控制论路线):
- 正确: Transformer是动力系统, token是概率流控制算子
- 过于简化: 对Qwen3/GLM4来说"控制算子"太简单(k90=38-62, 高度情境化)
- 但对DS7B正确: DS7B的否定确实接近1维控制算子(k90=2)

分析二(SVD优先, 批判性):
- 正确: SVD分析是最关键的实验 — 现在有了!
- 正确: k>50意味着无统一结构 — Qwen3/GLM4确实如此
- 但错误: 不能一概而论 — DS7B的k90=2证明某些模型确实有简洁结构
- 最关键的洞察: **否定的数学结构依赖于模型**, 不存在通用结构

### 理论进展: 否定结构的模型依赖性

已确定的数学性质:
1. 否定不是对合: T_not(T_not(P)) ≠ P (Phase 235确认)
2. 否定是加法机制: h_not(P) = h(P) + Δ_not(context) (Phase 235确认)
3. **否定的维度是模型依赖的** (Phase 236新发现!)
   - DS7B: 近1维 (logit空间top1=81.6%)
   - Qwen3: 中高维 (logit空间top1=45.0%)
   - GLM4: 高维 (logit空间top1=13.7%)
4. DS7B在中间层有1维否定瓶颈 (k90=1, top1=97.5%)

数学形式的修正:
- 对DS7B: Δ_not = α · d_not (几乎1维!) + small noise
- 对Qwen3: Δ_not = Σ_{i=1}^{38} α_i · d_i (中高维)
- 对GLM4: Δ_not = Σ_{i=1}^{62} α_i · d_i (高维)

### 硬伤与问题

1. 句子多样性不足: 所有句子都是"X is Y"格式, 不同句型可能改变SVD结果
2. DS7B低维vs简化: DS7B可能是更简洁的表示(好), 也可能是能力不足(坏)
3. 跨模型维度差异的原因未明: 架构? 训练数据? 模型规模?
4. 子空间重叠可能主要来自位置效应(所有算子插在同一位置), 非语义效应
5. 未测试更复杂的否定(条件否定, 作用域否定等)

### 第一性原理分析

**核心洞察**: 语言模型对否定的表示存在"维度谱" — 从DS7B的1维到GLM4的62维.
这挑战了"否定有通用数学结构"的假设. 不同模型用不同维度解决了同一个问题.

但DS7B的低维结构提供了突破路径:
1. 如果DS7B的1维否定方向可以被提取和理解, 我们就能定义"程序基元"
2. DS7B可以作为"最简模型"来研究, 然后检验更复杂模型是否包含相同基元
3. 如果Qwen3的前2个logit SVD方向与DS7B的1维方向对齐, 说明"简洁核心"被高维噪声包裹

**突破瓶颈的方向**: 研究DS7B的1维否定方向
- 它在logit空间压制/增强哪些token?
- 它与Qwen3的top-1 logit方向是否对齐?
- 在DS7B上做1维steering是否有效?

这些实验可以直接回答"语言模型内部的否定程序是什么"这个核心问题.

### 下一步方向

1. 提取DS7B的1维否定方向 — 在logit空间定义否定基元
2. 跨模型方向对齐 — DS7B的1维方向 vs Qwen3的top-k方向
3. 1维steering实验 — 在DS7B上注入否定方向, 测试概率操控
4. 更多句型测试 — 验证低维结构是否鲁棒
5. 架构归因 — 为什么DS7B学到了1维否定而其他模型没有?

### 测试脚本

- tests/glm5/phase236_program_geometry.py
- 结果: tests/glm5_temp/phase236_{qwen3,glm4,deepseek7b}_results.json
"""

# Write back
with open(memo_path, "w", encoding="utf-8") as f:
    f.writelines(before)
    f.write(new_content)

# Verify
with open(memo_path, "r", encoding="utf-8") as f:
    new_lines = f.readlines()

print(f"Total lines after: {len(new_lines)}")
# Check Phase 236 section
for i, line in enumerate(new_lines):
    if "Phase 236" in line:
        print(f"Phase 236 at line {i+1}: {line.rstrip()[:80]}")
        for j in range(i, min(i+5, len(new_lines))):
            print(f"  L{j+1}: {new_lines[j].rstrip()[:80]}")
        break
print("Done! Garbled text replaced with correct Chinese.")
