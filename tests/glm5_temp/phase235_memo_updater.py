"""Phase 235 MEMO更新脚本"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

memo = """
## Phase 235: Negation Symmetries & Mechanisms [2026-05-19 23:45]

### 核心目标

回应分析二的关键批判:
1. Logit Lens的rho(early)约等于0.03可能是投影噪声
2. Value@L0 cos=1.0是数学必然不是发现
3. 需要从现象描述转向寻找数学不变量和对称性

4个实验:
- ExpA: 双重否定对称性 T_not(T_not(P)) 约等于 I ?
- ExpB: 门控vs加法 否定是乘法还是加法机制?
- ExpC: Position-Aligned Activation Patching 因果层定位
- ExpD: 否定vs形容词对比 Logit Lens低rho是否否定特有?

### 核心发现1: 双重否定不恢复肯定句 (三模型一致!)

T_not(T_not(P)) 不等于 I

| 模型 | KL ratio (notnot/P vs not/P) | corr(aff,notnot) | corr(aff,not) | cos(aff,notnot) | cos(aff,not) |
|------|------------------------------|------------------|---------------|-----------------|--------------|
| Qwen3-4B | 7.86 | 0.842 | 0.959 | 0.897 | 0.979 |
| GLM4-9B | 1.96 | 0.795 | 0.886 | 0.771 | 0.889 |
| DS7B | 3.51 | 0.941 | 0.940 | 0.958 | 0.958 |

关键解读:
- KL ratio >> 1: notnot P比not P离P更远! 双重否定不是恢复而是进一步偏离
- cos(aff,notnot) < cos(aff,not): Qwen3和GLM4中否定句反而比双重否定句更接近肯定句
- 否定不是对合(involution): T_not不是可逆算子

Type A (结构性: It is not true that X is not Y) vs Type B (形态学: not un-Y):
| 模型 | Type A KL ratio | Type B KL ratio |
|------|-----------------|-----------------|
| Qwen3 | 14.36 | 1.37 |
| GLM4 | 2.23 | 1.13 |
| DS7B | 4.06 | 1.88 |

Type B接近恢复(KL ratio 1.1-1.9), Type A严重偏离(KL ratio 2-14).
原因: Type A改变了句子结构(增加了It is not true that前缀), Type B只修改了前缀(not un- vs un-).

数学意义:
- 否定算子T_not不满足对合律: T_not(T_not(P)) 不等于 P
- 这排除了否定是简单逻辑取反的可能
- 语言模型中的否定不是布尔逻辑中的NOT

### 核心发现2: 否定是加法机制 (Qwen3/GLM4确认)

| 模型 | ratio跨句cos | diff跨句cos | 判定 |
|------|-------------|------------|------|
| Qwen3 | 0.054 | 0.267 | ADDITIVE |
| GLM4 | 0.034 | 0.235 | ADDITIVE |
| DS7B | 0.790 | 0.745 | MIXED (slightly ratio) |

Qwen3和GLM4: diff跨句cosine远大于ratio跨句cosine (5-7倍), 说明:
h_not(P) 约等于 h(P) + Delta_not(context)
其中Delta_not是上下文依赖但形状一致的偏移量.

DS7B: ratio和diff都较高且接近, 说明DS7B的否定机制更复杂(可能既有加法也有乘法成分).

这否定了分析二提出的门控假设(至少对Qwen3/GLM4):
h_not(P) 不等于 g(context) * h(P)
因为如果门控成立, ratio应该比diff更稳定.

### 核心发现3: Logit Lens低rho是投影噪声 (三模型一致!)

| 模型 | neg rho(early) | adj rho(early) | diff | neg rho(late) | adj rho(late) |
|------|---------------|----------------|------|---------------|---------------|
| Qwen3 | 0.033 | 0.023 | -0.010 | 0.789 | 0.774 |
| GLM4 | 0.012 | 0.013 | +0.001 | 0.710 | 0.706 |
| DS7B | 0.015 | 0.034 | +0.019 | 0.027 | 0.203 |

关键结论:
- 否定和形容词的rho(early)非常接近, 差异仅0.01-0.02
- 这意味着Phase 234的分布式重计算结论中, rho(early)约等于0.03这一部分很可能是Logit Lens的投影噪声, 不是否定特有的
- GLM4和DS7B中形容词的rho(early)甚至高于否定!
- DS7B的rho(late)也极低(0.027 vs Qwen3的0.789), 说明DS7B的Logit Lens在所有层都不太有效

对Phase 234结论的修正:
Phase 234认为rho(early)约等于0.03是否定的分布式重计算证据. 但ExpD证明这是Logit Lens在早期层的普遍问题(对形容词也是0.02-0.03). 因此, rho(early)低不能作为否定特有机制的证据.

然而, Phase 234的ExpA增量残差分解(Incr_KL 约等于 Cum_KL)仍然有效, 因为那个实验不依赖Logit Lens.

### 核心发现4: 操作符位置的因果层定位

| 模型 | op_pos最因果层 | op_pos KL | last_pos最因果层 | last_pos KL |
|------|---------------|-----------|-----------------|-------------|
| Qwen3 | L4 | 0.27 | L0 | 5.19 |
| GLM4 | L0 | 0.50 | L0 | 7.42 |
| DS7B | L0 | 6.03 | L3 | 6.60 |

Qwen3独特: L4是操作符位置最因果关键的层, 之后效果递减. 这意味着在Qwen3中, 否定信号在L4从not token位置开始向其他位置传播.

GLM4/DS7B: L0就是最因果关键的, 说明否定信息从第一层就在not位置存在(这是预期的, 因为embedding不同).

逐层模式(Qwen3 op_pos):
- L0: KL=0.13 (embedding差异)
- L4: KL=0.27 (峰值! 否定信号在此层最强)
- L8-L35: 递减 (信号已传播到其他位置)

### 对两份分析的综合评判

分析一(支持性): 方向正确但不够深入. 建议的Logit Flow分解和程序基元提取是有价值的, 但需要先解决Logit Lens的有效性问题.

分析二(批判性): 大部分批判被实验证实:
1. Value@L0 cos=1.0确实是数学必然 - 正确
2. Logit Lens低rho可能是投影噪声 - 被ExpD证实!
3. 需要找不变量和对称性 - 正确, 但双重否定测试显示T_not不是对合
4. 门控假设 - 被ExpB否定(Qwen3/GLM4是否定是加法不是门控)

但分析二也有过于悲观的地方:
- 虽然T_not(T_not(P)) 不等于 I, 但Type B双重否定接近恢复(KL ratio 1.1-1.9)
- 否定是加法机制这一发现本身就是一个具体的数学性质

### 硬伤与问题

1. ExpC的GLM4/DS7B结果需要更多等长句对验证
2. DS7B的ExpB结果是混合的, 需要更深入分析
3. 双重否定测试中Type A的句子结构差异是混淆因素
4. 没有测试更复杂的否定组合(如三重否定, 条件否定等)
5. 缺少与逻辑否定(0->1)的对照, 无法区分语义否定vs逻辑否定

### 理论进展: 否定的数学性质

已确定的数学性质:
1. T_not不是对合: T_not(T_not(P)) 不等于 P
2. T_not是加法型: h_not(P) 约等于 h(P) + Delta_not(context)
3. T_not的Delta_not跨句子形状一致但幅度变化
4. T_not在Qwen3中于L4涌现因果效应

否定程序的候选数学形式:
Delta_logit_l = f_l(h_l, context)
P_not = P_aff + sum_l Delta_logit_l

其中f_l是层特定的加法修正函数, 每层独立贡献.

下一步关键问题:
1. Delta_not的形状是什么? 有多少个自由度?
2. 不同控制算子(not/never/if/but)的Delta是否共享结构?
3. 否定在logit空间的作用模式是什么(suppress/boost/redirect)?

### 下一步方向

1. 否定Delta的SVD分解: Delta_not = sum alpha_i * u_i, 找到主要成分
2. 跨控制算子对比: not vs never vs if vs but的Delta模式
3. 非线性Steering: 既然否定是加法机制, 尝试Delta注入(加法steering)而非方向steering
4. 条件否定的结构: 不同上下文下Delta_not的变化模式
5. 定义程序基元: 从logit空间提取suppress/boost/redirect操作
"""

with open('research/glm5/docs/AGI_GLM5_MEMO.md', 'a', encoding='utf-8') as f:
    f.write(memo)
print('MEMO updated successfully')
