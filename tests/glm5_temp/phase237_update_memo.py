"""Update AGI_GLM5_MEMO.md with Phase 237 results"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

memo_path = "research/glm5/docs/AGI_GLM5_MEMO.md"

new_content = """
## Phase 237: Negation Primitive Decoding — d_not 的语义身份 [2026-05-20 02:40]

### 核心目标

从"DS7B有一个1维否定方向"到"这个方向在语义上是什么"——从几何描述到编码机制理解的跳跃

### 实验设计

- ExpA: 提取d_not + logit空间解码 — d_not在token层面做什么?
- ExpB: 否定行为测试 — 低维是能力还是缺陷?
- ExpC: 多句型鲁棒性 — 1维结构是否跨句型成立?
- ExpD: 跨模型方向对齐 — 三模型是否共享核心方向?

### 核心发现1: DS7B的低维否定是缺陷,不是能力! (最重要的发现!)

| 模型 | 简单否定准确率 | 蕴含准确率 | 总体准确率 |
|------|--------------|-----------|-----------|
| Qwen3 | 72.7% | 100% | 81.8% |
| GLM4 | 77.3% | 90.9% | 81.8% |
| **DS7B** | **27.3%** | 100% | **51.5%** |

DS7B的简单否定判断准确率27.3% — 远低于随机水平50%!
DS7B不能可靠区分肯定句和否定句.

这意味着Phase 236发现的"DS7B中间层1维否定瓶颈"不是优雅的数学结构,
而是模型容量不足导致的粗暴压缩. DS7B把否定压缩到1维, 丢失了关键语义.

### 核心发现2: Qwen3的d_not编码了"确认/验证"语义

Qwen3 logit d_not的Top boosted tokens:
- 正确(correct), 正面(positive), 否(no), 相符(matching)
- 解析(parse), 好的(good), 成功(success), 不错(not bad)
- 符合(conform), 证明(prove), 前提是(premise)

Top suppressed tokens:
- Need, Unable, Worse, Bad, Massive

**语义解读**: 否定方向增加了"正确/验证/符合"类token的概率,
减少了"需要/更差/坏"类token. 否定操作使模型进入"验证/确认"模式,
而不是简单的"翻转极性".

这与Phase 235的发现一致: 否定不是逻辑NOT(极性翻转), 而是概率重塑.

### 核心发现3: DS7B的d_not是语义空壳

DS7B logit d_not的Top boosted tokens:
- Utf, intl, .${, ".., 在玩家, Executors, RVA — 全是代码/格式噪声!

没有任何否定语义相关的token. DS7B的1维方向虽然在几何上存在,
但在语义上是空的 — 它压缩了否定的统计相关性, 但丢失了语义内容.

### 核心发现4: GLM4的d_not有不同策略

GLM4 logit d_not的Top boosted tokens:
- odynam, Erdo, NASA, Mayor, Marin — 专有名词!
Top suppressed tokens:
- 这个, 这句话, 这是一个 — 中文指示词!

GLM4的策略: 否定增强专有名词/特定实体, 抑制指代词.
这与Qwen3的"验证模式"不同 — GLM4的否定更接近"实体聚焦".

### 核心发现5: 跨模型方向对齐是假象

| 模型对 | Logit cosine |
|--------|-------------|
| Qwen3 vs DS7B | 0.91 |
| Qwen3 vs GLM4 | -0.49 |
| GLM4 vs DS7B | -0.45 |

Qwen3 vs DS7B的0.91高cosine是假象! 原因:
1. 150K维度中99%+是低幅噪声, 这些维度主导cosine计算
2. 即使移除top-5000最高幅度维度, cosine仍然0.91
3. 但真正有意义的top维度(语义token)完全不同
4. Qwen3的d_not编码"验证语义", DS7B的d_not编码噪声

**结论: cosine相似度在超大规模logit空间中是不可靠的对齐指标!**

### 核心发现6: 低维结构跨句型鲁棒,但仍是缺陷

| 句型 | DS7B k90 | Qwen3 k90 | GLM4 k90 |
|------|----------|-----------|----------|
| 系动词(X is Y) | 4 | 29 | 32 |
| 行为动词(X runs) | 5 | 29 | 34 |
| 量词(All X) | 3 | 31 | 32 |
| 信念(I think) | 4 | 26 | 29 |

DS7B的低维结构在所有句型中都成立(k90=3-5),
Qwen3/GLM4在所有句型中都是高维(k90=26-34).
这说明DS7B的压缩是通用策略, 不是"X is Y"句型的捷径.

### 对两份分析的综合评判

分析一(几何控制论):
- 错误: 把DS7B的1维瓶颈解读为"抽象压缩" — 实际是容量缺陷
- 正确: Transformer是动力系统, 否定是概率重塑
- 正确: Qwen3的d_not确实编码了语义(验证/确认模式)
- 过于乐观: "DS7B找到了最简抽象表示" — 不, DS7B丢失了语义

分析二(SVD优先,批判性):
- **完全正确**: "低维≠简洁结构" — DS7B就是最好的证据!
- **完全正确**: 需要行为测试验证 — 现在做了, 结果明确
- **完全正确**: 子空间重叠可能是假象 — cosine在高维空间不可靠
- **完全正确**: 需要解码d_not的语义 — 现在做了, Qwen3有语义,DS7B没有
- 正确但需补充: "多句型测试" — 低维跨句型鲁棒,但仍是缺陷

### 理论进展: 否定编码的三层结构

基于Phase 235-237的系统发现, 否定在Transformer中的编码有三层:

1. **几何层**: Δ_not ≈ Σ α_i · d_i (加法机制, Phase 235)
2. **维度层**: 自由度k90=26-62 (高维, Phase 236) 
3. **语义层**: d_not编码"验证/确认"模式 (Phase 237新发现!)

DS7B只有第1层和部分第2层, 缺失第3层 — 所以行为失败.

### 硬伤与问题

1. 否定行为测试样本太少(33题), 且Yes/No格式可能不够敏感
2. d_not解码受双语tokenizer影响 — Qwen3的d_not主要激活中文token
3. 跨模型cosine对齐在高维空间不可靠 — 需要更好的对齐方法
4. 未区分"否定增强验证语义"vs"否定本身就是验证操作"的因果方向
5. GLM4的"实体聚焦"策略需要更多验证

### 第一性原理分析

**核心洞察**: 否定不是极性翻转,而是"验证模式激活".

当模型处理否定时, 它不是在翻转"黑→白",
而是在激活一组"验证/确认/比较"token的预测概率.
这使得"正确/符合/证明"类token更可能, "需要/更差/坏"类token更不可能.

这解释了为什么双重否定不恢复原句:
- 第一次否定: 激活验证模式
- 第二次否定: 在已验证的状态上再次验证 → 不是回到原点, 而是叠加验证

这也解释了为什么否定不是对合:
- 验证(验证(P)) ≠ P — 连续验证产生的是"元验证"状态, 不是原状态

**更深层**: 语言的"否定"可能对应一种**注意力重分配操作**:
否定 = 从"直接预测"模式切换到"比较/验证"模式.
这不是1维控制信号, 而是高维的概率流重定向.

### 下一步方向

1. **在Qwen3上做d_not steering实验** — 注入d_not, 测试是否能让模型进入"验证模式"
2. **区分因果方向** — 是否定导致验证模式, 还是验证模式是否定的表征?
3. **研究更多控制算子的d_token** — always/never/if/because的d_token是什么语义?
4. **改进跨模型对齐方法** — 用语义token集合而非cosine来比较
5. **回到"结构压缩机制"** — Qwen3的40维logit表示中, 哪些维度是核心?

### 测试脚本

- tests/glm5/phase237_negation_primitive.py
- 结果: tests/glm5_temp/phase237_{qwen3,glm4,deepseek7b}_results.json
"""

with open(memo_path, 'a', encoding='utf-8') as f:
    f.write(new_content)

print('MEMO updated successfully')
