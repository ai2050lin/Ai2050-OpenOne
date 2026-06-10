"""更新MEMO: Phase 438 + 综合结论"""
import os
from datetime import datetime

memo_path = "research/glm5/docs/AGI_GLM5_MEMO.md"
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

content = f"""

## Phase 438: 运输算子跨对象迁移 [{timestamp}]

### 实验目标
验证类别运输方向是否可以在同类对象间迁移

### 方法
1. 计算src对象的fruit运输方向(注入fruit方向后的delta)
2. 将该delta注入tgt对象的对应层
3. 测量tgt的类别logit变化
4. transfer_score = src_cat_delta - opp_cat_delta

### 关键结果 (同类迁移, best layer, beta=2.0)

#### Qwen3
| 迁移对 | transfer_score |
|--------|---------------|
| apple->orange | 0.11 |
| apple->lemon | 0.32 |
| dog->cat | -0.01 |
| dog->horse | 0.12 |
| knife->hammer | **0.95** |
| knife->spoon | **0.53** |
| car->train | 0.31 |
| car->bus | 0.27 |

#### GLM4
| 迁移对 | transfer_score |
|--------|---------------|
| apple->orange | **0.82** |
| apple->lemon | **0.87** |
| dog->cat | 0.30 |
| dog->horse | 0.24 |
| knife->hammer | **3.37** |
| knife->spoon | **3.14** |
| car->train | -0.02 |
| car->bus | 0.11 |

#### DS7B
| 迁移对 | transfer_score |
|--------|---------------|
| apple->orange | **-0.21** |
| apple->lemon | **-0.50** |
| dog->cat | -0.08 |
| dog->horse | -0.32 |
| knife->hammer | -0.40 |
| knife->spoon | 0.09 |
| car->train | 0.11 |
| car->bus | 0.05 |

### 客观现象
1. **Qwen3**: 正transfer，tool类最强(knife->hammer=0.95)
2. **GLM4**: 正transfer更强(knife->hammer=3.37!)，fruit类也有效(0.82-0.87)
3. **DS7B**: 几乎全部为负或近零，运输方向不跨对象共享

### 严格审视
硬伤1: 跨类别迁移全部为空(可能NaN)，缺少关键对照
硬伤2: GLM4的强transfer与Phase 437的弱mediation矛盾:
  - 运输方向可以在同类对象间迁移
  - 但类别改变不导致属性改变
  → GLM4有类别级运输方向，但属性不依赖类别

---

## Phase 434-438 综合结论与理论修正 [{timestamp}]

### 核心发现矩阵

| 维度 | Qwen3 | GLM4 | DS7B |
|------|-------|------|------|
| 单头因果贡献 | 极低(分布式) | 混合 | 未验证 |
| 上下文化属性方向 | 存在,cos(WE)≈0 | 存在,cos(WE)≈0 | 数值问题 |
| 最后一层属性注入 | 有效(sw=2-9) | 有效(sw=2-9) | NaN |
| 属性-类别中介 | **强(4.75-6.44)** | **弱/近零** | **弱(0.3-1.1)** |
| 同类运输迁移 | 正(0.1-0.95) | **强(0.3-3.4)** | **负(-0.5~+0.1)** |

### 最重要的修正

1. **类别-属性中介不是通用机制！**
   - Qwen3: 属性由类别中介，推类别→属性跟着变
   - GLM4: 属性不由类别中介，推类别→属性不变
   - DS7B: 弱/混合中介

2. **GLM4的矛盾发现**
   - 运输方向在同类对象间可迁移(Phase 438: 3.37)
   - 但类别改变不影响属性(Phase 437: -0.04)
   → GLM4有类别级运输方向但属性独立于类别

3. **运输是分布式过程**
   - 单头消融几乎无影响
   - 类别运输由多个头共同完成
   - 没有单一的"路由头"

4. **上下文化属性方向与静态方向正交**
   - cos(contextual, W_E) ≈ 0 for ALL models and layers
   - 属性信息在上下文化过程中被重新编码
   - 最后一层注入有效但中间层不稳定

### 对用户分析的修正

用户说:
- "注意力头负责位置路由" → 修正: 没有单一头关键，运输是分布式的
- "属性是二阶、关系槽位条件化因子" → 修正: 这是Qwen3的情况，
  GLM4中属性是独立因子，DS7B中是弱中介
- "自然运输方向是更接近因果方向" → 仍然成立，但运输是分布式过程

### 理论升级

最新理论必须加入"模型特异性"维度:

```
语言编码的数学结构可能不是唯一的:
- Qwen3: 层级编码 (category → property mediation strong)
- GLM4: 独立编码 (category transport exists, but properties independent)
- DS7B: 弱结构 (neither category mediation nor transport transfer)
```

这挑战了"语言有统一数学结构"的假设。
不同训练策略/架构/数据可能导致不同的内部编码方式。

### 瓶颈分析

当前最大瓶颈:
1. 对象数量仍然不足(每类2-3个对象)
2. 跨类别迁移测试失败(全部为空)
3. DS7B的数值稳定性问题
4. 属性-类别中介的模型差异无法在当前框架下解释

### 突破方向

1. 扩大对象集(每类10-20个)以区分"类别通用"和"对象特定"
2. 用更多属性维度(color, taste, material, part, shape, size)验证中介
3. 分析GLM4为什么属性不依赖类别——可能GLM4采用了对象-属性直接绑定
4. 在不同模板上测试(不仅是"An X is a kind of")
5. 比较不同训练数据/架构对编码方式的影响
"""

with open(memo_path, "a", encoding="utf-8") as f:
    f.write(content)

print(f"MEMO updated at {timestamp}")
