# -*- coding: utf-8 -*-
import io

MEMO = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'

section = """

## Phase 2874：属性轴词表扩容 v2（零前向 3.9s，Gen1→Gen2 两轮）

**日期**：2026-09-18。**脚本**：`tests/glm5/phase2874_attr_vocab_v2.py`。
**产物**：`phase2874/attr_vocab_v2/` {execution.json `06ecd0ef`,
result.json `b07eecd1`, attr_vocab_v2.npz `5f7796e1`}。

### 原理

修复 2873 诊断的词数功效瓶颈：每属性轴扩到 ≥5 个 single-token 极性词，
用 2858 式几何判据链（VG1 覆盖 / VG2a 轴向正交 / VG2b 同轴对再现 / VG3
范数健康）在**花任何前向预算之前** gate 词表。判据预注册冻结于任何
unembed 统计之前（execution.json 先行）。

### Gen1→Gen2

Gen1 池（weightless/feeble/luminous/arid 四词非单 token）VG1=false
（6/10 轴）。按 2857 CAND2 补足先例扩池（ponderous/mighty/radiant/soggy
换入），**门线一字不动**，清旧产物后重跑。

### Gen2 判决表

| 判据 | 观测 | 门线 | 结果 |
|---|---|---|---|
| **VG1** 覆盖 | **8/10 轴** ≥5 词（weight 4、moisture 3 不足） | ≥8 轴 | **true** |
| **VG2a** 轴向正交 | max 跨轴 cos **0.1869** | <0.488 | **true** |
| **VG2b** 同轴对再现 | 均值 **0.2731**（8 轴） | >0.2 | **true**（2870 同轴 0.46-0.59，v2 池更抽象故略降） |
| **VG3** 范数健康 | 最差轴比 **1.2039** | <3.0 | **true** |
| **vocab_legal** | 8 轴 / **42 词** / 26 对 | — | **True** |

隔离轴：weight（ponderous 仍非单 token）、moisture（soggy 非单 token）。

### 接续

Phase 2875 = 42 词全谱 census（2872 协议逐字移植 + X5 功效复检：42 词
LOO-NN 检索 vs 全管线镜像 null）；随后 2876 = 增长率第二点重测（2873
协议跑在新词表上）。
"""

with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(section)
print('APPENDED')
