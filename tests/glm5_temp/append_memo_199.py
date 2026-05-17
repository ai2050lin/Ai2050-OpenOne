"""Append Phase 199 results to AGI_GLM5_MEMO.md"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

memo_content = r'''

## Phase 199: Syntax-Controlled Semantic Perturbation + Delay Spectrum Mapping [2026-05-16 23:55]

### 核心目标

在Phase 198"纠错阶段"确认KL slope是tautology之后，本阶段解决两个关键问题：

1. **语义效应 vs 句法混淆**：之前的question/negation/conditional效应中，多少是真正的语义信号，多少是句法结构变化造成的伪象？
2. **延迟谱深度映射**：不同语言结构影响未来的时间尺度是否真有系统性差异？
3. **模式跃迁连续性**：模式切换是离散相变还是连续变化？

### 实验设计

**Exp1: Syntax-Controlled Semantic Perturbation** (最关键)

| 条件 | 类型 | 示例 |
|------|------|------|
| sem_negation | 语义+句法 | "The cat does not chase the dog" |
| syn_negation | 纯句法 | "The cat then chase the dog" |
| syn_insertion | 无意义插入 | "The cat zzz chase the dog" |
| sem_question | 语义+句法 | "Does the cat chase the dog?" |
| syn_question | 纯句法 | "Therefore, the cat chase the dog" |
| syn_question_v2 | 纯标点 | "The cat chases the dog?" |
| sem_conditional | 语义+句法 | "If the cat chases the dog" |
| syn_conditional | 纯句法 | "When the cat chases the dog" |
| syn_conditional_v2 | 纯句法 | "After the cat chases the dog" |

**Exp2: Delay Spectrum Deep Mapping**

从即时约束到超长程约束的完整延迟谱：
- 即时: negation, question
- 延迟: conditional_if, conditional_unless, future_will, past_tense
- 长程: narrative_setup ("In a world where..."), suppose_that
- 基线: rand_token, rand_period

**Exp3: Mode Transition Continuity**

渐变prompt路径：
- CoT: "Problem" → "Think" → "Think carefully" → "Think step by step" → "Let's think step by step"
- Translation: "The cat sleeps" → "in Chinese:" → "Translate" → "Translate to Chinese:"
- Coding: "Add two numbers" → "using code:" → "Write code to" → "def add_two_numbers"

### 数据量

| 模型 | 句子数 | 步数 | 采样数 |
|------|--------|------|--------|
| Qwen3 | 20 | 12 | 30 |
| GLM4 | 8 | 8 | 15 |
| DS7B | 8 | 8 | 15 |

### 核心结果

#### Exp1: 语义 vs 句法 KL[0] 对照表

| 对照组 | Qwen3 Delta | Qwen3 语义? | GLM4 Delta | GLM4 语义? | DS7B Delta | DS7B 语义? |
|--------|-------------|-------------|------------|-----------|-----------|-----------|
| sem_neg vs syn_neg (then) | +0.801 | MARGINAL | +0.800 | MARGINAL | +0.939 | MARGINAL |
| sem_neg vs syn_insertion (zzz) | +0.531 | MARGINAL | -0.566 | MARGINAL | -2.439 | YES(反转!) |
| sem_q vs syn_question (Therefore,) | +10.384 | **YES** | +6.198 | **YES** | +13.231 | **YES** |
| sem_q vs syn_question_v2 (只加?) | +2.584 | **YES** | -0.346 | MARGINAL | +2.271 | **YES** |
| sem_cond vs syn_cond (When) | -0.538 | MARGINAL | -0.561 | MARGINAL | +0.719 | MARGINAL |
| sem_cond vs syn_cond_v2 (After) | -0.660 | MARGINAL | -0.793 | MARGINAL | -0.372 | MARGINAL |

#### Exp1 关键发现

**1. Question效应是真实的语义效应（三模型一致）**

- sem_question vs syn_question: Delta = +10.4/+6.2/+13.2 → **三模型都YES**
- "Does X?" vs "Therefore, X" — 即使两者都改变了句法，疑问语序+助动词倒装的真实疑问效应远超纯句法重构
- 但：syn_question_v2（只加问号不改变语序）的KL[0]=8.1/6.7/11.4 → **标点本身就能产生很大KL**
- 结论：question效应 = 语义疑问 + 句法重构 + 标点效应的**叠加**，但语义疑问部分是真实的

**2. Negation效应可能是句法效应为主（三模型一致）**

- sem_negation vs syn_negation: Delta仅+0.8/+0.8/+0.9 → MARGINAL
- "does not" vs "then" 在KL[0]上几乎没有差别！
- 更严重的是DS7B中 syn_insertion("zzz")的KL[0]=4.12 > sem_negation的1.68
- **这暗示negation的KL[0]主要来自token插入造成的序列位移，而非语义否定**
- 这是一个重要发现：否定可能不是"约束"，而是"扰动"

**3. Conditional效应：语义效应微弱，If/When/After几乎等价**

- sem_conditional vs syn_conditional: Delta = -0.54/-0.56/+0.72 → MARGINAL
- If vs When vs After的KL[0]几乎相同 → 它们主要共享"从句结构"效应
- **conditional的"延迟效应"可能是从句结构本身的属性，不是"If"特有的假设世界构建**

#### Exp2: 延迟谱跨模型对比

| 条件 | Qwen3 KL[0] | Qwen3 Delay | GLM4 KL[0] | GLM4 Delay | DS7B KL[0] | DS7B Delay |
|------|-------------|-------------|------------|-----------|-----------|-----------|
| negation | 1.16 | 0 (IMM) | 1.24 | 0 (IMM) | 1.68 | 0 (IMM) |
| question | 10.70 | 0 (IMM) | 6.38 | 0 (IMM) | 13.63 | 0 (IMM) |
| conditional_if | 0.46 | 1 (FAST) | 0.35 | 1 (FAST) | 1.65 | 0 (IMM) |
| conditional_unless | 0.86 | 1 (FAST) | 0.84 | 1 (FAST) | 2.44 | 0 (IMM) |
| future_will | 0.37 | 1 (FAST) | 0.35 | 1 (FAST) | 1.05 | 0 (IMM) |
| past_tense | 0.34 | 1 (FAST) | 0.24 | 1 (FAST) | 0.63 | 1 (FAST) |
| narrative_setup | 5.22 | 0 (IMM) | 5.01 | 0 (IMM) | 6.87 | 0 (IMM) |
| suppose_that | 0.42 | 1 (FAST) | 0.27 | 1 (FAST) | 0.77 | 1 (FAST) |
| rand_token | 0.80 | 1 (FAST) | 2.07 | 0 (IMM) | 3.41 | 0 (IMM) |
| rand_period | 9.10 | 0 (IMM) | 7.06 | 0 (IMM) | 7.11 | 0 (IMM) |

#### Exp2 关键发现

**1. DS7B的conditional_if不显示延迟！KL[0]=1.65（IMMEDIATE）**

- Qwen3/GLM4: conditional_if KL[0]≈0.35-0.46 (FAST_DELAY)
- DS7B: conditional_if KL[0]=1.65 (IMMEDIATE)
- 这意味着DS7B中"If"立即产生了比Qwen3/GLM4更大的初始发散
- **可能的解释**：DS7B（DeepSeek-R1-Distill-Qwen-7B）是蒸馏模型，可能对条件句的处理方式不同

**2. 延迟效应可能只是"弱初始发散"的另一个名字**

- conditional_if的KL[0]=0.46(Qwen3), 但rand_token的KL[0]=0.80 → **rand_token的KL[0]也"不大"**
- 关键问题：conditional的KL[0]小，究竟是因为"延迟"，还是因为"If"只是在第0步碰巧没改变概率分布？
- 如果是后者，"延迟"只是弱初始效应的自然结果，不是真正的时间非对称性

**3. 即时/延迟分界线：KL[0] > 1.0 → IMMEDIATE, KL[0] < 1.0 → FAST_DELAY**

- 这个分界很可能是trivial的：任何KL[0] > 1的算IMMMEDIATE，< 1的算FAST_DELAY
- rand_token KL[0]=0.80(Qwen3)也落入了FAST_DELAY → **延迟分类可能没有语义意义**

#### Exp3: 模式跃迁连续性

| 模式 | Qwen3 max_kl_jump | Qwen3 连续? | GLM4 max_kl_jump | GLM4 连续? | DS7B max_kl_jump | DS7B 连续? |
|------|-------------------|------------|-------------------|-----------|-------------------|-----------|
| CoT | 13.97 | **NO** | 11.12 | **NO** | 12.91 | **NO** |
| Translation | 2.40 | YES | 4.31 | **NO** | 4.27 | **NO** |
| Coding | 19.81 | **NO** | 8.03 | **NO** | 2.63 | YES |

#### Exp3 关键发现

**1. CoT是强烈的离散相变（三模型一致）**

- 从"Think carefully"到"Think step by step"出现巨大KL跳跃（Qwen3: 0.53→14.50）
- 这说明"step by step"不是渐进式推理增强，而是突然切换到完全不同的计算模式
- CoT确实是一种"模式切换"，而非"推理增强"

**2. Translation的不一致：Qwen3连续，GLM4/DS7B不连续**

- Qwen3中翻译模式是连续过渡的（max_kl_jump=2.40）
- GLM4/DS7B中翻译出现不连续跳跃（4.31/4.27）
- 这可能反映了不同模型对翻译的内部表示方式不同

**3. Coding在DS7B中连续但在Qwen3/GLM4中不连续**

- Qwen3: 从"Write code to"到"def"跳跃19.8 → "def"是一个强触发器
- DS7B: 渐变更平滑（max_kl_jump=2.63）
- 编程模式可能在不同模型中有不同的触发机制

### 理论分析

#### A. 修正Phase 198的结论

Phase 198认为conditional delayed effect是"最可能真机制"。Phase 199对此进行了严格检验：

**修正1**: Conditional的"延迟"可能只是"弱初始发散"

- If/When/After的KL[0]几乎相同(0.35-1.65) → 它们的主要效应是"从句结构"，不是假设世界
- conditional_if vs rand_token的delay差异仅为0-1步 → 统计上不显著
- 但conditional的slope确实较大(Qwen3: 1.12 vs rand_token: 0.92) → 需要更精细的分析

**修正2**: Question效应是真实的，但包含大量句法成分

- sem_question vs syn_question的Delta=+6到+13 → 真实语义信号
- 但syn_question_v2(只加问号)也有KL[0]=6.7-11.4 → 标点贡献巨大
- 需要进一步：纯语义疑问(不加问号不改语序)的测试

**修正3**: Negation效应可能主要来自token序列位移

- sem_negation vs syn_negation(then)几乎无差别 → 否定不是语义约束
- 这是一个重大修正：否定可能不是"约束"，而只是"扰动"
- 但注意：negation的attractor basin与syn_negation不同 → 分流结构可能不同

#### B. 三层结构的修正

Phase 198的三层结构基本正确，但需要细化：

**Level 1: Mode (离散相变)**
- CoT是真正的相变（三模型一致）
- Translation和Coding的连续性因模型而异
- 模式不是"连续流形"上的区域，而更像"离散吸引子"间的跳跃
- 但CoT的强相变可能因为"step by step"是一个特别强的触发器

**Level 2: Constraint (需要重新定义)**
- Negation不是"约束"→ 是"扰动"
- Question不是"约束"→ 是"模式切换"(切换到QA模式)
- Conditional可能是"弱扰动"而非"延迟约束"
- 真正的"约束"可能需要满足：改变允许的token集但不改变整体生成模式

**Level 3: Autoregressive Chaos (确认)**
- Phase 198的结论完全正确：KL slope是tautology
- Phase 199进一步确认：即使是语义条件，KL slope也落入同一范围

#### C. 当前最硬的问题

1. **Conditional delay是真的还是弱的伪象?**
   - 如果delay_step只差0-1步，且与rand_token没有显著差异
   - 那么"conditional延迟效应"可能只是"初始扰动较小"的自然推论
   - 需要更大数据量和更严格的统计检验

2. **Negation是否是"约束"?**
   - 如果negation与随机token插入在KL[0]上无显著差异
   - 那么否定可能只是序列扰动，不是语义约束
   - 但attractor basin的差异可能暗示：虽然初始发散相似，但分流结构不同

3. **"语义效应"vs"句法效应"的严格区分是否可能?**
   - 任何语义操作都必然伴随句法变化
   - 问题是：如何量化"纯语义"效应?
   - 可能的方向：同义改写(paraphrase) — 不变语义但变token

### 脚本

- tests/glm5/phase199_syntax_delay.py

### 日志

- tests/glm5_temp/phase199_qwen3_log.txt (316.5s)
- tests/glm5_temp/phase199_glm4_log.txt (2333.5s)
- tests/glm5_temp/phase199_ds7b_log.txt (1490.9s)

### 结果JSON

- tests/glm5_temp/phase199_qwen3_results.json
- tests/glm5_temp/phase199_glm4_results.json
- tests/glm5_temp/phase199_deepseek7b_results.json
'''

# Append to MEMO
memo_path = r"D:\Ai2050\TransformerLens-Project\research\glm5\docs\AGI_GLM5_MEMO.md"
with open(memo_path, 'a', encoding='utf-8') as f:
    f.write(memo_content)
print(f"Appended Phase 199 results to {memo_path}")
