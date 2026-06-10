"""更新MEMO: Phase 434-437"""
import os
from datetime import datetime

memo_path = "research/glm5/docs/AGI_GLM5_MEMO.md"
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

content = f"""

## Phase 434: 注意力头因果消融 [{timestamp}]

### 实验目标
验证Phase 431的候选routing heads是否真正搬运类别信息

### 方法
1. 计算原始自然运输方向 delta_last = h_last(perturbed) - h_last(clean)
2. 消融候选头: 将该头输出置零
3. 消融后重新计算 delta_last_ablated
4. 因果分数 = 1 - ||delta_ablated|| / ||delta_orig||

### 关键结果

#### Qwen3 (n_heads=32, head_dim=80)
| 对象 | 候选头CausalScore均值 | 控制头CausalScore均值 | gap |
|------|---------------------|---------------------|-----|
| apple | +0.007 | +0.185 | -0.179 |
| dog | +0.022 | -0.006 | +0.028 |
| knife | -0.002 | -0.206 | +0.204 |
| car | +0.025 | -0.092 | +0.117 |

#### GLM4 (n_heads=32, head_dim=128)
| 对象 | 候选头CausalScore均值 | 控制头CausalScore均值 | gap |
|------|---------------------|---------------------|-----|
| apple | -0.198 | +0.007 | -0.205 |
| dog | +0.219 | +0.256 | -0.037 |
| knife | +0.070 | -0.130 | +0.199 |
| car | -0.108 | -0.240 | +0.133 |

#### DS7B (n_heads=28, head_dim=128)
所有CausalScore = 0.000 (ablation hook未生效)

### 客观现象
1. 单头消融因果分数极低，候选头和控制头无清晰区分
2. Qwen3: CausalScore < 0.1，候选头甚至弱于控制头(apple)
3. GLM4: 混合结果，L3/H17对apple有-0.47(反向增强)，但跨对象不一致
4. DS7B: ablation hook在Qwen2架构上未正确工作

### 严格审视
硬伤1: 单头消融对delta_norm的影响极小，说明类别运输是分布式过程
硬伤2: DS7B的ablation hook可能需要不同的实现方式
硬伤3: 需要多头联合消融或path patching才能真正验证因果
硬伤4: 候选头选择基于Phase 431的attention weight，但高attn weight不等于类别搬运

---

## Phase 436: 上下文化属性方向 [{timestamp}]

### 实验目标
测试属性信息是否存在于上下文化的hidden states中（而非静态embedding）

### 方法
1. 构造属性对比句对: "The color of the apple is red." vs "...green."
2. 前向传播两个句子，提取各层last token的hidden state
3. 计算上下文化属性方向: d_attr = h(red_ctx) - h(green_ctx)
4. 将方向注入到测试模板对应层
5. 与静态W_E属性方向对比

### 关键结果

#### 上下文化方向与静态方向的余弦
| 模型 | cos(contextual, W_E) | cos(contextual, W_U) |
|------|---------------------|---------------------|
| Qwen3 | -0.01 ~ +0.05 | -0.01 ~ +0.05 |
| GLM4 | -0.02 ~ +0.01 | -0.01 ~ +0.01 |
| DS7B | -0.04 ~ +0.02 | -0.04 ~ +0.17 |

上下文化属性方向与静态W_E/W_U方向几乎正交！

#### 最后一层注入效果
| 属性 | Qwen3 L35 neg_sw | GLM4 L39 neg_sw |
|------|------------------|-----------------|
| apple/color | 2.500 | 5.865 |
| dog/color | 5.500 | 4.678 |
| apple/taste | 1.953 | 3.120 |
| knife/material | **8.094** | **9.464** |
| car/part | 0.000 | 0.000 |

#### 中间层注入效果
- switch_score经常为负（方向反转）
- 效果不稳定，层间波动大

#### DS7B数值问题
- 8bit量化导致所有logits为NaN
- 上下文化方向范数极端大(L6=630, L12=839)

### 客观现象
1. 上下文化属性方向确实存在，但与静态W_E/W_U几乎正交
2. 最后一层注入有效(switch=2-9)，但中间层混乱
3. neg_injection（注入反方向）比pos_injection效果更一致
4. car/part的neg_injection switch=0，说明某些属性方向不可操控

### 严格审视
硬伤1: 最后一层注入效果可能只是直接修改读出，不是"操控内部表示"
硬伤2: 中间层注入不稳定说明方向在层间被重新编码
硬伤3: DS7B的8bit量化严重影响实验
硬伤4: 属性方向可能包含除了"属性"以外的其他信息（句子结构差异等）

---

## Phase 437: 属性是否由类别中介 [{timestamp}]

### 实验目标
测试改变类别轨道后，属性是否跟着变

### 方法
1. 用category方向在embedding层将对象从源类别推向目标类别
2. 测量属性词logit变化
3. mediation_score = tgt_props_delta - src_props_delta
4. 正值 = 属性跟随类别变化

### 关键结果 (alpha=2.0)

#### Qwen3: 强正mediation
| 推方向 | src_props_delta | tgt_props_delta | mediation |
|--------|----------------|----------------|-----------|
| apple: fruit->animal | **-3.15** | **+1.61** | **+4.75** |
| apple: fruit->tool | **-2.28** | **+4.01** | **+6.29** |
| knife: tool->vehicle | **-2.98** | **+3.46** | **+6.44** |
| dog: animal->fruit | **-2.40** | **+3.88** | **+6.28** |
| car: vehicle->tool | +0.85 | +2.69 | +1.84 |

#### GLM4 (bf16): 近零/负mediation
| 推方向 | src_props_delta | tgt_props_delta | mediation |
|--------|----------------|----------------|-----------|
| apple: fruit->animal | -0.44 | -0.48 | -0.04 |
| apple: fruit->tool | -0.39 | -0.35 | +0.04 |
| knife: tool->vehicle | +0.06 | -0.04 | -0.10 |
| dog: animal->fruit | +0.04 | +0.04 | -0.00 |
| car: vehicle->tool | +0.01 | **-1.39** | **-1.40** |

#### DS7B (bf16): 弱/混合mediation
| 推方向 | src_props_delta | tgt_props_delta | mediation |
|--------|----------------|----------------|-----------|
| apple: fruit->animal | +0.09 | +0.87 | +0.78 |
| apple: fruit->tool | -0.03 | +0.31 | +0.33 |
| knife: tool->vehicle | +1.42 | +1.37 | -0.05 |
| dog: animal->fruit | -0.27 | -0.73 | -0.46 |
| car: vehicle->tool | +0.95 | +1.68 | +0.74 |

### 客观现象
1. **Qwen3: 属性确实由类别中介！** 类别切换时属性跟随变化(mediation=4.75-6.44)
2. **GLM4: 属性不由类别中介！** bf16结果确认这不是量化问题
3. **DS7B: 弱/混合中介** 部分方向有正mediation但远弱于Qwen3
4. car->tool在Qwen3中mediation最低(+1.84)，在GLM4中最负(-1.40)

### 严格审视
硬伤1: 模型间差异巨大——类别-属性中介不是通用机制
硬伤2: GLM4中category push方向可能不够有效（类别logit变化也小）
硬伤3: alpha=0.5和1.0时mediation很弱，说明需要大扰动才能看到效果
硬伤4: src_props在GLM4和DS7B中也变化了，但方向不一致

### 关键洞察
类别-属性中介是模型特异的结构，不是语言编码的通用机制。
Qwen3可能采用了"类别→属性"的层级编码策略，
而GLM4可能采用了"对象→属性"的直接绑定策略。
这意味着语言编码的数学结构在不同模型中可能不同！

---

## Phase 434-437 综合结论 [{timestamp}]

### 最可靠的结论
1. 单头消融对类别运输影响极小 → 类别运输是分布式过程
2. 上下文化属性方向存在但与静态W_E/W_U正交 → 属性编码不是线性方向
3. 最后一层注入属性方向有效但中间层不稳定 → 属性信息在深层被重新编码
4. 属性-类别中介在Qwen3中强(mediation=4-6)，在GLM4中弱/负，DS7B混合 → 模型特异

### 对用户分析的修正
1. **用户说"注意力头负责路由"过于简单** — 单头消融证明无单一头关键，运输是分布式的
2. **用户说"属性是二阶因子"需要修正** — 属性在Qwen3中确实由类别中介(二阶)，
   但在GLM4中属性可能独立于类别(独立因子)，模型间差异巨大
3. **用户说"自然运输方向更接近因果方向"仍然成立** — 但运输是分布式的，非单头负责

### 理论升级方向
最新理论必须加入"模型特异性"维度：
- Qwen3: 类别→属性层级编码，强中介，线性可操控
- GLM4: 对象→属性直接绑定，弱中介，8bit/bf16一致
- DS7B: 混合策略，弱中介，数值稳定性差

这意味着"语言编码的数学结构"可能不是唯一的！
不同训练策略/架构/数据可能导致不同的内部编码方式。
"""

with open(memo_path, "a", encoding="utf-8") as f:
    f.write(content)

print(f"MEMO updated at {timestamp}")
